import argparse
import os
import sys
from datetime import datetime, timedelta, date

import es as es_module
from feed.parse_feeds import DEFAULT_TIMEZONE, export_from_elastic, parse_and_export


def parse_day(value: str) -> datetime | None:
    """Resolve a --day value. "all" means no day filter at all."""
    v = value.strip().lower()
    now = datetime.now()
    if v == "all":
        return None
    if v == "today":
        return now
    if v == "yesterday":
        return now - timedelta(days=1)
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.combine(date.fromisoformat(value), datetime.min.time())


def main():
    parser = argparse.ArgumentParser(description="News crawler with NoDCore export")
    parser.add_argument(
        "cmd",
        choices=["scrape", "nod"],
        help="scrape: fetch feeds (and index to Elasticsearch). "
             "nod: rebuild a day's corpus from Elasticsearch.",
    )
    # Defaults come from the environment so the container is a drop-in for the
    parser.add_argument(
        "--feeds-file",
        default=os.environ.get("CRAWLER_FEEDS_FILE", "data/feeds_de.txt"),
        help="Path to feeds file [env: CRAWLER_FEEDS_FILE]",
    )
    parser.add_argument(
        "--lang",
        default=os.environ.get("CRAWLER_LANGUAGE", "german"),
        help="Language name; also selects the news-<lang> index [env: CRAWLER_LANGUAGE]",
    )
    parser.add_argument(
        "--timezone",
        default=os.environ.get("CRAWLER_TIMEZONE", DEFAULT_TIMEZONE),
        help="Zone the --day filter is evaluated in [env: CRAWLER_TIMEZONE]",
    )
    parser.add_argument(
        "--out-dir",
        default=os.environ.get("CRAWLER_OUT_DIR", "out/nod"),
        help="Output directory root for .bz2 files [env: CRAWLER_OUT_DIR]",
    )
    parser.add_argument(
        "--day",
        default="yesterday",
        help="Which day to process: yesterday, today, an ISO date like 2026-01-26, "
             "or 'all' to take whatever the feeds currently list (scrape only)",
    )
    parser.add_argument(
        "--content-source",
        choices=["article", "feed"],
        default="article",
        help='Extract sentences from fetched article HTML ("article") or from feed summary ("feed")',
    )
    parser.add_argument(
        "--no-corpus",
        action="store_true",
        help="Skip writing the .bz2; only index to Elasticsearch. Used by the "
             "hourly cron job, which leaves corpus building to the nod command.",
    )
    parser.add_argument(
        "--no-elastic",
        action="store_true",
        help="Ignore ELASTIC_URL and run without Elasticsearch",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    args = parser.parse_args()
    target_day = parse_day(args.day)

    indexer = None if args.no_elastic else es_module.from_env(args.lang)
    if indexer is not None:
        try:
            version = indexer.ping()
        except es_module.ElasticError as e:
            print(f"Elasticsearch at {indexer.url} is unreachable: {e}", file=sys.stderr)
            return 1
        if args.verbose:
            print(f"Elasticsearch {version} at {indexer.url} -> index {indexer.articles_index}")

    if args.cmd == "nod":
        if indexer is None:
            print(
                "The nod command rebuilds the corpus from Elasticsearch, "
                "so ELASTIC_URL must be set.",
                file=sys.stderr,
            )
            return 1
        created_files = export_from_elastic(
            indexer=indexer,
            day=target_day,
            lang=args.lang,
            out_root=args.out_dir,
            time_zone=args.timezone,
            verbose=args.verbose,
        )
    else:
        created_files = parse_and_export(
            feeds_file=args.feeds_file,
            lang=args.lang,
            out_root=args.out_dir,
            day=target_day,
            content_source=args.content_source,
            time_zone=args.timezone,
            indexer=indexer,
            write_corpus=not args.no_corpus,
            verbose=args.verbose,
        )

    print("\n✓ Pipeline complete!")
    print(f"  Created {len(created_files)} .bz2 file(s)")
    for f in created_files:
        print(f"    {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
