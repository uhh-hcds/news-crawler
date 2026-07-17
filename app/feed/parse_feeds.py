from __future__ import annotations

import re
import bz2
from dataclasses import dataclass
from datetime import datetime, date, timezone
from pathlib import Path
from typing import List, Optional, Tuple, Literal
import socket
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from html import unescape
from zoneinfo import ZoneInfo

import feedparser
import nltk

import es as es_module


USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64; rv:57.0) Gecko/20100101 Firefox/57.0"

ContentSource = Literal["article", "feed"]

DEFAULT_TIMEZONE = "Europe/Berlin"
DEFAULT_LANGUAGE = "german"

# Kept out of out/nod/<lang>, which NoDCore wipes after each import.
DEFAULT_FAILURES_ROOT = "out/feeds"

# The crawler's language names already match NLTK's; map the short codes the
# CLI also accepts.
_LANGUAGE_ALIASES = {"de": "german", "en": "english"}


def strip_html_tags(text: str) -> str:
    """Remove HTML tags and clean text"""
    if not text:
        return ""
    
    text = unescape(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&\w+;', ' ', text)
    text = re.sub(r'style\s*=\s*["\'][^"\']*["\']', '', text)
    text = re.sub(r'class\s*=\s*["\'][^"\']*["\']', '', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

_DROP_TAGS_RE = re.compile(
    r"(?is)<(script|style|noscript|svg|form|footer|nav|header|aside)[^>]*>.*?</\1>"
)


def extract_article_text(html: str) -> str:
    """
    Heuristic extraction of main article text from HTML.

    """
    if not html:
        return ""

    html = _DROP_TAGS_RE.sub(" ", html)

    def _best_container(tag: str) -> str:
        matches = re.findall(rf"(?is)<{tag}\b[^>]*>(.*?)</{tag}>", html)
        if not matches:
            return ""
        best = ""
        best_len = 0
        for m in matches:
            cleaned = strip_html_tags(m)
            cleaned_len = len(cleaned)
            if cleaned_len > best_len:
                best_len = cleaned_len
                best = cleaned
        return best

    # Prefer semantic containers if present
    article = _best_container("article")
    if article:
        return article.strip()

    main = _best_container("main")
    if main:
        return main.strip()

    # Otherwise, join paragraph text (common across news sites)
    paragraphs = re.findall(r"(?is)<p\b[^>]*>(.*?)</p>", html)
    parts = []
    for p in paragraphs:
        t = strip_html_tags(p)
        if len(t.split()) < 4:
            continue
        parts.append(t)

    if parts:
        return "\n".join(parts).strip()

    return strip_html_tags(html).strip()


def split_into_sentences(text: str, language: str = DEFAULT_LANGUAGE) -> List[str]:
    """Split text into sentences with NLTK's Punkt tokenizer.

    """
    if not text:
        return []

    language = _LANGUAGE_ALIASES.get(language, language)
    try:
        sentences = nltk.sent_tokenize(text, language=language)
    except LookupError:
        # No Punkt model for this language; fall back to the default rather
        # than losing the document entirely.
        sentences = nltk.sent_tokenize(text)

    return [s.strip() for s in sentences if s.strip()]


def normalize_sentence(text: str) -> str:
    """Collapse whitespace, remove pipes, and neutralise backslashes.

    Whitespace collapsing also strips tabs and newlines, which is what keeps
    each corpus line to exactly one tab.

    Backslashes are dropped because NoDCore bulk-loads the derived TSVs with
    `LOAD DATA LOCAL INFILE` (Importer.scala:66) using the default
    `FIELDS ESCAPED BY '\\'`. A backslash that survives into an entity name is
    un-escaped at load time -- a literal backslash-t becomes a real TAB inside
    the value, shifting every subsequent column left and writing a row with
    name='Foo<TAB>2019-04-03', date='0000-00-00' and useEntity holding
    dayFrequency's value. 
    """
    text = " ".join(text.split())
    text = text.replace("|", " ")
    return text.replace("\\", " ")


# U+FFFD is what a failed decode leaves behind. NoDCore's fix_encoding pass
# deletes any line containing one, so they are dead weight at best.
_REPLACEMENT_CHAR = "\ufffd"

# UTF-8 read as latin-1/cp1252 leaves tell-tale pairs: "Ã¤" for ä, "â€œ" for a
# quote. NoDCore's fix_encoding repairs a fixed list of these with sed;
# anything not on that list reaches the corpus as garbage.
_MOJIBAKE_RE = re.compile(
    "[\u00c3\u00c2][\u0080-\u00bf]"   # \u00c3\u00a4 for a-umlaut, \u00c2 + nbsp
    "|\u00e2\u20ac"                     # smart quotes, dashes, ellipsis
    "|\u00c3\u00af\u00c2\u00bf\u00c2\u00bd"  # doubly-encoded U+FFFD
)

# Control characters, tab and newline included. Callers normalise whitespace
# away first, so this only fires when that was skipped -- which is exactly when
# it needs to. NoDCore reads every corpus line with
# `val Array(sentence, source) = line.split(tab)`: a line carrying anything
# other than exactly one tab raises MatchError and takes the whole import down,
# so a single bad sentence would cost a day's data.
_CONTROL_RE = re.compile(r"[\x00-\x1F\x7F]")


def has_bad_encoding(text: str) -> bool:
    """True if the text shows signs of a botched decode.

    Excluded rather than repaired: a sentence that already reads as mojibake
    carries no information worth guessing at, and letting it through risks
    taking NoDCore's import down with it.
    """
    if _REPLACEMENT_CHAR in text:
        return True
    if _CONTROL_RE.search(text):
        return True
    if _MOJIBAKE_RE.search(text):
        return True
    return False


def should_keep(text: str) -> bool:
    """Filter sentences by length (20-256 characters) and encoding sanity."""
    if not 20 <= len(text) <= 256:
        return False
    if has_bad_encoding(text):
        return False
    return True


@dataclass
class SentenceRecord:
    """A single sentence with metadata"""
    text: str
    url: str
    day: date
    
    def to_tuple(self) -> Tuple[str, str, str]:
        """Convert to tuple format (text, url, day_str)"""
        return (self.text, self.url, self.day.strftime("%Y%m%d"))


def load_feed_urls(path: str | Path) -> List[str]:
    """Load feed URLs from file"""
    feed_path = Path(path)
    if not feed_path.is_absolute():
        feed_path = (Path.cwd() / feed_path).resolve()
    
    lines = feed_path.read_text(encoding="utf-8").splitlines()
    urls: List[str] = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    return urls


def _fetch_bytes(url: str, *, timeout_s: int = 20) -> bytes:
    """Fetch URL content as bytes"""
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout_s) as resp:
        return resp.read()

_META_CHARSET_RE = re.compile(rb"""charset["'\s]*=["'\s]*([a-zA-Z0-9_-]+)""", re.I)


def _meta_charset(raw: bytes) -> Optional[str]:
    """Charset declared in a <meta> tag, which often contradicts the header."""
    match = _META_CHARSET_RE.search(raw[:4096])
    return match.group(1).decode("ascii", "ignore") if match else None


def _decode_html(raw: bytes, declared: Optional[str]) -> str:
    """Decode a page body, trying real candidates before giving up.

    """
    candidates = ["utf-8"]
    for candidate in (declared, _meta_charset(raw)):
        if candidate and candidate.lower() not in candidates:
            candidates.append(candidate.lower())
    candidates += ["cp1252", "latin-1"]

    for encoding in candidates:
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


def _fetch_text(url: str, *, timeout_s: int = 60) -> str:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read()
        declared = getattr(resp.headers, "get_content_charset", lambda: None)()
        return _decode_html(raw, declared)


def _entry_published(entry) -> Optional[datetime]:
    """Feed timestamp as an aware UTC datetime.

    """
    parsed = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if not parsed:
        return None
    return datetime(*parsed[:6], tzinfo=timezone.utc)


def _entry_guid(entry, article_url: str) -> str:
    """Stable document id, falling back to the URL."""
    return str(getattr(entry, "id", "") or "") or article_url


def parse_feeds_to_sentences(
    feeds_file: str | Path = "data/feeds_de.txt",
    *,
    day: Optional[datetime] = None,
    content_source: ContentSource = "article",
    time_zone: str = DEFAULT_TIMEZONE,
    lang_name: str = DEFAULT_LANGUAGE,
    indexer: Optional["es_module.Indexer"] = None,
    verbose: bool = False,
) -> Tuple[List[SentenceRecord], List[str]]:
    """
    Parse RSS/Atom feeds and extract sentence records directly.

    Args:
        feeds_file: Path to file with feed URLs (one per line)
        day: Only keep items published on this day (optional)
        content_source: Extract from article HTML ("article") or feed summary/content ("feed")
        time_zone: Zone the `day` filter is evaluated in (feed timestamps are UTC)
        lang_name: Language for the Punkt sentence tokenizer
        indexer: Optional Elasticsearch indexer; articles already present are
            not downloaded again, which is what makes hourly runs cheap
        verbose: Print progress information

    Returns:
        (sentence_records, failed_feed_urls)
    """
    sources = load_feed_urls(feeds_file)
    tz = ZoneInfo(time_zone)

    if verbose:
        print(f"Loaded {len(sources)} feed URLs")

    target_date = day.date() if day else None
    sentences: List[SentenceRecord] = []
    failed: List[str] = []
    feed_failures = 0
    article_failures = 0
    skipped = 0

    for i, url in enumerate(sources, 1):
        if verbose and i % 10 == 0:
            print(f"Processing feed {i}/{len(sources)}...")

        try:
            raw = _fetch_bytes(url)
            parsed = feedparser.parse(raw)
            entries = getattr(parsed, "entries", []) or []

            # Pick the entries for the target day before touching the network,
            # so the guid lookup below can be done in one batch per feed.
            candidates = []
            for entry in entries:
                published = _entry_published(entry)
                if not published:
                    continue
                if target_date and published.astimezone(tz).date() != target_date:
                    continue
                article_url = str(getattr(entry, "link", "") or "")
                if not article_url:
                    continue
                candidates.append((entry, published, article_url, _entry_guid(entry, article_url)))

            # Anything Elasticsearch already holds needs no second download.
            known = set()
            if indexer is not None and candidates:
                known = indexer.known_guids([c[3] for c in candidates])

            for entry, published, article_url, guid in candidates:
                if guid in known:
                    skipped += 1
                    continue

                article_date = published.astimezone(tz).date()

                content = ""
                if content_source == "article":
                    try:
                        html = _fetch_text(article_url)
                        content = extract_article_text(html)
                    except (HTTPError, URLError, TimeoutError, socket.timeout) as e:
                        failed.append(f"article_fetch\t{article_url}\t{e}")
                        article_failures += 1
                        if indexer is not None:
                            indexer.add_failure(
                                message=str(e),
                                url=article_url,
                                status=getattr(e, "code", 0) or 0,
                                time=datetime.now(timezone.utc),
                            )
                        continue
                    except Exception as e:
                        failed.append(f"article_fetch\t{article_url}\t{e}")
                        article_failures += 1
                        if indexer is not None:
                            indexer.add_failure(
                                message=str(e),
                                url=article_url,
                                status=0,
                                time=datetime.now(timezone.utc),
                            )
                        continue

                    if not content:
                        failed.append(f"article_extract\t{article_url}\tempty")
                        article_failures += 1
                        if indexer is not None:
                            indexer.add_failure(
                                message="Extractor returned no content",
                                url=article_url,
                                status=0,
                                time=datetime.now(timezone.utc),
                            )
                        continue
                else:
                    # Fall back to the feed-provided summary/content
                    content_obj = getattr(entry, "content", None)
                    if content_obj:
                        try:
                            content = content_obj[0].value
                        except Exception:
                            content = ""
                    if not content:
                        content = str(getattr(entry, "summary", "") or "")
                    content = strip_html_tags(content)

                if indexer is not None:
                    indexer.add_article(
                        title=str(getattr(entry, "title", "") or ""),
                        content=content,
                        url=article_url,
                        published=published,
                        guid=guid,
                    )

                for sent in split_into_sentences(content, language=lang_name):
                    normalized = normalize_sentence(sent)
                    if should_keep(normalized):
                        sentences.append(
                            SentenceRecord(text=normalized, url=article_url, day=article_date)
                        )

        except Exception as e:
            if verbose:
                print(f"Failed to fetch {url}: {e}")
            failed.append(f"feed_fetch\t{url}\t{e}")
            feed_failures += 1

    if indexer is not None:
        indexer.flush()

    if verbose:
        ok_feeds = max(0, len(sources) - feed_failures)
        print(f"\nExtracted {len(sentences)} sentences from {ok_feeds} feeds")
        print(f"Failures: {len(failed)} (feeds: {feed_failures}, articles: {article_failures})")
        if indexer is not None:
            print(
                f"Elasticsearch: indexed {indexer.indexed}, "
                f"skipped {skipped} already-indexed, logged {indexer.failures_logged} failures"
            )

    return sentences, failed


def write_nod_corpus(
    sentences: List[SentenceRecord],
    lang: str,
    out_root: Path = Path("out/nod"),
    verbose: bool = False,
) -> List[Path]:
    """
    Write sentences to NoDCore format: one YYYYMMDD.bz2 file per day.
    
    Args:
        sentences: List of sentence records
        lang: Language code (de, en, etc.)
        out_root: Output directory root
        verbose: Print progress information
    
    Returns:
        List of created file paths
    """
    out_dir = out_root / lang
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Group sentences by day
    by_day = {}
    for rec in sentences:
        by_day.setdefault(rec.day, []).append(rec)
    
    if verbose:
        print(f"\nWriting to {out_dir}")
        print(f"Sentences across {len(by_day)} day(s):")
        for day in sorted(by_day.keys()):
            print(f"  {day}: {len(by_day[day])} sentences")
    
    created_files = []

    for day, records in by_day.items():
        filename = day.strftime("%Y%m%d") + ".bz2"
        out_path = out_dir / filename

        written = 0
        dropped = 0
        with bz2.open(out_path, "wt", encoding="utf-8", compresslevel=2) as f:
            for rec in records:
                # collapse whitespace, filter length,
                # replace pipes. Collapsing whitespace also removes any tab or
                # newline inside the sentence, which is what guarantees each
                # line has exactly one tab.
                text = " ".join(rec.text.split())
                text = text.replace("|", " ")
                # Re-checked here rather than trusting the caller: this is the
                # last gate before NoDCore, and it reads every line with
                # `val Array(sentence, source) = line.split(tab)`, which throws
                # MatchError on a malformed line and aborts the whole import.
                if not should_keep(text) or not rec.url:
                    dropped += 1
                    continue
                f.write(f"{text}\t{rec.url}\n")
                written += 1

        created_files.append(out_path)

        if verbose:
            suffix = f", dropped {dropped}" if dropped else ""
            print(f"  Wrote {out_path.name} ({written} sentences{suffix})")

    return created_files


def store_failures(
    failed_urls: List[str],
    lang: str,
    failures_root: str | Path = DEFAULT_FAILURES_ROOT,
) -> None:
    """Store failed feed URLs to a log file.

    """
    if not failed_urls:
        return

    out_dir = Path(failures_root) / lang
    out_dir.mkdir(parents=True, exist_ok=True)

    failures_path = out_dir / "failures.log"
    with failures_path.open("a", encoding="utf-8") as f:
        for url in failed_urls:
            f.write(url + "\n")


def parse_and_export(
    feeds_file: str | Path = "data/feeds_de.txt",
    lang: str = "de",
    out_root: str | Path = "out/nod",
    day: Optional[datetime] = None,
    content_source: ContentSource = "article",
    time_zone: str = DEFAULT_TIMEZONE,
    failures_root: str | Path = DEFAULT_FAILURES_ROOT,
    indexer: Optional["es_module.Indexer"] = None,
    write_corpus: bool = True,
    verbose: bool = False,
) -> List[Path]:
    """
    Complete pipeline: parse feeds, optionally index to Elasticsearch, and
    export to NoDCore format.

    Args:
        feeds_file: Path to feeds file
        lang: Language name; drives both the output folder and the tokenizer
        out_root: Output directory
        day: Filter by day (optional)
        time_zone: Zone the `day` filter is evaluated in
        failures_root: Where failures.log goes; kept out of out_root because
            NoDCore deletes everything under its content mount
        indexer: Optional Elasticsearch indexer
        write_corpus: Write the .bz2. Hourly runs that only top up
            Elasticsearch pass False, leaving the corpus to `nod`, which
            rebuilds the whole day at once.
        verbose: Print progress

    Returns:
        List of created .bz2 files
    """
    sentences, failed = parse_feeds_to_sentences(
        feeds_file=feeds_file,
        day=day,
        content_source=content_source,
        time_zone=time_zone,
        lang_name=lang,
        indexer=indexer,
        verbose=verbose,
    )

    created_files: List[Path] = []
    if write_corpus:
        created_files = write_nod_corpus(
            sentences=sentences,
            lang=lang,
            out_root=Path(out_root),
            verbose=verbose,
        )

    # Store failures
    if failed:
        store_failures(failed, lang, failures_root)
        if verbose:
            print(f"\nLogged {len(failed)} failures to failures.log")

    return created_files


def export_from_elastic(
    indexer: "es_module.Indexer",
    day: datetime,
    lang: str,
    out_root: str | Path = "out/nod",
    time_zone: str = DEFAULT_TIMEZONE,
    verbose: bool = False,
) -> List[Path]:
    """
    Rebuild a day's NoD corpus from Elasticsearch.

    """
    articles = indexer.articles_for_day(day, time_zone)
    if verbose:
        print(f"Loaded {len(articles)} articles from {indexer.articles_index} for {day:%Y-%m-%d}")

    target_day = day.date()
    sentences: List[SentenceRecord] = []
    for article in articles:
        url = article.get("url") or ""
        content = article.get("content") or ""
        if not url or not content:
            continue
        for sent in split_into_sentences(content, language=lang):
            normalized = normalize_sentence(sent)
            if should_keep(normalized):
                sentences.append(SentenceRecord(text=normalized, url=url, day=target_day))

    return write_nod_corpus(
        sentences=sentences,
        lang=lang,
        out_root=Path(out_root),
        verbose=verbose,
    )


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parents[1]
    created = parse_and_export(
        feeds_file=BASE_DIR / "data" / "feeds_de.txt",
        lang="de",
        out_root="out/nod",
        day=datetime.now(),
        verbose=True
    )
    
    print(f"\n✓ Created {len(created)} file(s)")
    
    if created:
        print(f"\nSample from {created[0].name}:")
        with bz2.open(created[0], "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    text, url = parts
                    print(f"  {text[:60]}...")
                    print(f"  {url}\n")
