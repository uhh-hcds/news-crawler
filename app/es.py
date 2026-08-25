"""Elasticsearch indexing for the news crawler.

ES Index  for the new  crawler

    news-<lang>      {title, content, url, published, guid, html, content_vector}
    failures-<lang>  {message, url, status, time}

Keyed by uid

ELASTIC_INDEX overrides the article index name for both this crawler and the
API's retriever, keeping the two ends from drifting apart; see from_env.


Talks to Elasticsearch through the official `elasticsearch` client (8.x, the
version pinned in api/requirements.txt), which is what core/retrieval.py
already uses against this same cluster.

Articles carry an mxbai-embed-large-v1 embedding in `content_vector` so
core/retrieval.py's ElasticRetriever can knn-search them. That field is the
reason news-<lang> is created from an explicit mapping instead of being left
to dynamic detection: dynamic mapping types a float array as `float`, not
`dense_vector`, and the field cannot be converted afterwards -- a reindex is
the only fix. failures-<lang> has no vector and is still created dynamically.

Embeddings are asymmetric: documents are encoded bare, queries are encoded
behind _QUERY_PREFIX. Using the wrong side silently degrades recall rather
than erroring, so both live here next to each other.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

from elasticsearch import (
    ApiError,
    BadRequestError,
    Elasticsearch,
    NotFoundError,
    TransportError,
)
from elasticsearch.helpers import BulkIndexError, bulk, scan
from sentence_transformers import SentenceTransformer

# Elasticsearch rejects oversized _bulk bodies; chunk well below the default
# 100MB http.max_content_length since article bodies are unbounded.
_BULK_CHUNK = 200

_EMBED_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
# Fallback only; the real value is read off the loaded model at index creation
# so a model swap cannot silently disagree with the mapping.
_EMBED_DIMS = 1024
_EMBED_BATCH = 32

# mxbai requires this exact prefix for searching to work accurately, and it
# must NOT be applied to documents. Same string as core/retrieval.py.
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# One model per process, not one per Indexer: a two-language run would
# otherwise hold two ~1.2GB copies of identical weights.
_MODELS: Dict[str, SentenceTransformer] = {}


def _get_model(name: str) -> SentenceTransformer:
    if name not in _MODELS:
        _MODELS[name] = SentenceTransformer(name)
    return _MODELS[name]


class ElasticError(RuntimeError):
    """Raised when Elasticsearch is configured but a request fails.

    `status` carries the HTTP code so callers can tell "index does not exist
    yet" (404, expected on a first run) from a real fault they must not
    silently ignore.
    """

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


def _wrap(context: str, e: Exception) -> ElasticError:
    """Translate a client exception into ElasticError, preserving the status.

    ApiError covers HTTP error responses, and carries the code on .meta.status;
    TransportError covers connection and timeout faults, which have no status.
    """
    status = getattr(getattr(e, "meta", None), "status", None)
    return ElasticError(f"{context} -> {e}", status=status)


@dataclass
class FeedItem:
    """One crawled article, shaped to match the news-<lang> mapping.

    Field names mirror the Go FeedItem. `content_vector` is filled in by the
    Indexer at flush time rather than at construction, so a caller never pays
    for an embedding on an article that turns out to be a duplicate.
    """

    title: str
    content: str
    url: str
    # The crawler hands this over as a datetime; the mapping wants a date, and
    # to_dict() bridges the two.
    published: Union[str, datetime]
    guid: str
    html: str = ""
    content_vector: Optional[List[float]] = None

    def to_dict(self) -> dict:
        """Converts the dataclass instance to a dictionary for Elasticsearch."""
        data = asdict(self)
        if isinstance(data["published"], datetime):
            data["published"] = data["published"].isoformat()
        # An explicit null in a dense_vector field is legal but pointless, and
        # it makes "was this embedded yet?" harder to answer downstream.
        if data["content_vector"] is None:
            del data["content_vector"]
        return data


def articles_mapping(dims: int = _EMBED_DIMS) -> Dict[str, Any]:
    """Mapping for news-<lang>, including the knn-searchable vector field."""
    return {
        "properties": {
            "title": {"type": "text"},
            "content": {"type": "text"},
            "url": {"type": "keyword"},
            # `date`, not `text`: articles_for_day() runs a range query with
            # `time_zone` and `format`, and Elasticsearch rejects both of those
            # on a text field.
            "published": {"type": "date"},
            "guid": {"type": "keyword"},
            "html": {"type": "text"},
            # Our dense vector field configuration
            "content_vector": {
                "type": "dense_vector",
                "dims": dims,
                "index": True,          # <-- Crucial! Enables fast native kNN search
                "similarity": "cosine",  # <-- Matches the embedding model metric
            },
        }
    }


class Indexer:
    """Buffered writer for the news-<lang> / failures-<lang> indices."""

    def __init__(
        self,
        url: str,
        lang: str,
        user: Optional[str] = None,
        password: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_s: int = 30,
        embedding_model: str = _EMBED_MODEL,
        index: Optional[str] = None,
    ):
        self.url = url.rstrip("/")
        self.lang = lang
        self.timeout_s = timeout_s
        # `index` overrides the derived name so the crawler and the API can be
        # pointed at one index by a single ELASTIC_INDEX env var; see from_env.
        # The failures index stays derived -- nothing on the API side reads it.
        self.articles_index = index or f"news-{lang}"
        self.failures_index = f"failures-{lang}"
        self.indexed = 0
        self.failures_logged = 0
        self.embedding_model = embedding_model
        self._index_ready = False

        # Basic auth wins over an API key when both are given, which is the
        # precedence the hand-rolled Authorization header had.
        auth: Dict[str, Any] = {}
        if user and password:
            auth["basic_auth"] = (user, password)
        else:
            key = api_key or os.getenv("ELASTIC_API_KEY")
            if key:
                auth["api_key"] = key

        self.client = Elasticsearch(
            self.url,
            request_timeout=timeout_s,
            # A crawl is long and unattended: ride out a brief node hiccup
            # instead of aborting the run.
            retry_on_timeout=True,
            max_retries=3,
            **auth,
        )
        self._actions: List[Dict[str, Any]] = []

    @property
    def model(self) -> SentenceTransformer:
        """The embedding model, loaded on first use.

        Lazy because the read paths (known_guids, articles_for_day) need no
        embeddings, and loading mxbai costs a ~1.2GB download on a cold cache.
        """
        return _get_model(self.embedding_model)

    def ping(self) -> str:
        """Return the Elasticsearch version, raising ElasticError if unreachable."""
        try:
            info = self.client.info()
        except (ApiError, TransportError) as e:
            raise _wrap("GET /", e) from e
        return str(info.get("version", {}).get("number", "unknown"))

    def close(self) -> None:
        """Release the underlying connection pool."""
        self.client.close()

    def __enter__(self) -> "Indexer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.flush()
        self.close()

    # -- embedding ---------------------------------------------------------

    def embed_document(self, title: str, content: str) -> List[float]:
        """Encode one article for storage. No prefix -- that is the query side."""
        return self.model.encode(f"{title} {content}").tolist()

    def embed_query(self, query: str) -> List[float]:
        """Encode a search string. Mirrors ElasticRetriever.embed_search_query."""
        return self.model.encode(_QUERY_PREFIX + query).tolist()

    def _embed_pending(self, actions: List[Dict[str, Any]]) -> None:
        """Fill in content_vector for every buffered article that lacks one.

        One encode() call per bulk chunk rather than one per article: batching
        is the whole reason to put a GPU behind this, and per-document calls
        leave it idle between articles.
        """
        pending = [
            a
            for a in actions
            if a["_index"] == self.articles_index
            and not a["_source"].get("content_vector")
        ]
        if not pending:
            return
        texts = [
            f"{a['_source'].get('title', '')} {a['_source'].get('content', '')}"
            for a in pending
        ]
        vectors = self.model.encode(texts, batch_size=_EMBED_BATCH)
        for action, vector in zip(pending, vectors):
            action["_source"]["content_vector"] = vector.tolist()

    # -- schema ------------------------------------------------------------

    def ensure_articles_index(self) -> None:
        """Create news-<lang> with the dense_vector mapping if it is missing.

        Only the article index needs this. dense_vector cannot be added by
        dynamic detection, and an index that has already typed content_vector
        as `float` needs a reindex, not a mapping update -- so this has to run
        before the first document lands.
        """
        if self._index_ready:
            return
        try:
            if not self.client.indices.exists(index=self.articles_index):
                dims = self.model.get_sentence_embedding_dimension() or _EMBED_DIMS
                self.client.indices.create(
                    index=self.articles_index, mappings=articles_mapping(dims)
                )
        except BadRequestError as e:
            # Two crawlers starting together both see "missing" and both
            # create; losing that race is fine, anything else is not.
            if "resource_already_exists" not in str(e):
                raise _wrap(f"PUT /{self.articles_index}", e) from e
        except (ApiError, TransportError) as e:
            raise _wrap(f"PUT /{self.articles_index}", e) from e
        self._index_ready = True

    # -- writing -----------------------------------------------------------

    def add_article(
        self,
        title: str,
        content: str,
        url: str,
        published: datetime,
        guid: str,
        html: str = "",
    ) -> None:
        """Queue an article. Matches the Go FeedItem field names."""
        self.add_feed_item(
            FeedItem(
                title=title,
                content=content,
                url=url,
                published=published,
                guid=guid or url,
                html=html,
            )
        )

    def add_feed_item(self, item: FeedItem) -> None:
        """Queue an already-built FeedItem.

        content_vector is left unset unless the caller supplied one; it gets
        filled in as a batch when the buffer is sent.
        """
        doc = item.to_dict()
        doc_id = item.guid or item.url
        doc["guid"] = doc_id
        self._actions.append(
            {
                "_op_type": "index",
                "_index": self.articles_index,
                "_id": doc_id,
                "_source": doc,
            }
        )
        self.indexed += 1
        self._maybe_send()

    def add_failure(
        self,
        message: str,
        url: str,
        status: int,
        time: datetime,
        guid: Optional[str] = None,
    ) -> None:
        """Queue a fetch failure.

        Keyed by guid so an hourly retry overwrites the previous attempt
        rather than appending a near-duplicate every hour.
        """
        doc_id = guid or url
        self._actions.append(
            {
                "_op_type": "index",
                "_index": self.failures_index,
                "_id": doc_id,
                "_source": {
                    "message": message,
                    "url": url,
                    "status": status,
                    "time": time.isoformat(),
                },
            }
        )
        self.failures_logged += 1
        self._maybe_send()

    def _maybe_send(self) -> None:
        if len(self._actions) >= _BULK_CHUNK:
            self._send_bulk()

    def _send_bulk(self) -> None:
        """Send buffered documents. Raises ElasticError if any item was rejected."""
        if not self._actions:
            return
        actions, self._actions = self._actions, []
        has_articles = any(a["_index"] == self.articles_index for a in actions)
        if has_articles:
            self.ensure_articles_index()
            self._embed_pending(actions)
        try:
            bulk(self.client, actions, chunk_size=_BULK_CHUNK)
        except BulkIndexError as e:
            reasons = []
            for item in e.errors:
                info = next(iter(item.values()), {})
                error = info.get("error")
                if error:
                    reasons.append(str(error.get("reason", error)))
            joined = "; ".join(reasons[:3])
            raise ElasticError(f"{len(reasons)} document(s) rejected: {joined}") from e
        except (ApiError, TransportError) as e:
            raise _wrap("POST /_bulk", e) from e

    def refresh(self) -> None:
        """Make freshly indexed documents searchable."""
        for index in (self.articles_index, self.failures_index):
            try:
                self.client.indices.refresh(index=index)
            except NotFoundError:
                # A refresh on an index this run never wrote to is not a fault.
                continue
            except (ApiError, TransportError) as e:
                raise _wrap(f"POST /{index}/_refresh", e) from e

    def flush(self) -> None:
        """Send anything buffered and make it searchable.

        Refresh happens once here rather than on every bulk chunk: per-chunk
        refreshes force a Lucene commit per few hundred docs and dominate the
        indexing cost.
        """
        self._send_bulk()
        self.refresh()

    # -- reading -----------------------------------------------------------

    def known_guids(self, guids: Sequence[str]) -> set:
        """Return the subset of guids already present in news-<lang>.

        Lets an hourly run skip re-downloading article HTML it already has.
        A missing index (first run) means nothing is known; any other error is
        raised, because treating an outage as "nothing is indexed" would
        silently re-download every article on every run.
        """
        found: set = set()
        if not guids:
            return found
        for i in range(0, len(guids), _BULK_CHUNK):
            chunk = list(guids[i : i + _BULK_CHUNK])
            try:
                result = self.client.mget(
                    index=self.articles_index, ids=chunk, source=False
                )
            except NotFoundError:
                return found
            except (ApiError, TransportError) as e:
                raise _wrap(f"POST /{self.articles_index}/_mget", e) from e
            for doc in result.get("docs", []):
                if doc.get("found"):
                    found.add(doc["_id"])
        return found

    def articles_for_day(
        self, day: datetime, time_zone: str, include_vector: bool = False
    ) -> List[Dict[str, Any]]:
        """Fetch every article published on `day`, interpreted in `time_zone`.

        Elasticsearch rounds a bare date down for gte and up for lte, so a single day string covers the
        whole day.

        content_vector is excluded by default: it is 1024 floats per document,
        which dwarfs the article text over a full day's scroll and is of no use
        to a corpus consumer.

        Only a missing index yields an empty result; other errors are raised
        so a transient outage cannot quietly produce an empty corpus that then
        overwrites a good one.
        """
        day_str = day.strftime("%Y-%m-%d")
        query = {
            "query": {
                "range": {
                    "published": {
                        "gte": day_str,
                        "lte": day_str,
                        "time_zone": time_zone,
                        "format": "yyyy-MM-dd",
                    }
                }
            }
        }
        articles: List[Dict[str, Any]] = []
        # scan() owns the scroll lifecycle, clearing the context when the
        # generator is exhausted -- a scroll context holds Lucene segments
        # open until it expires otherwise.
        hits: Iterator[Dict[str, Any]] = scan(
            self.client,
            index=self.articles_index,
            query=query,
            size=1000,
            scroll="2m",
            preserve_order=False,
            source_excludes=None if include_vector else ["content_vector"],
        )
        try:
            for hit in hits:
                articles.append(hit["_source"])
        except NotFoundError:
            return []
        except (ApiError, TransportError) as e:
            raise _wrap(f"POST /{self.articles_index}/_search", e) from e
        return articles

    def search_similar(
        self,
        query: str,
        k: int = 5,
        num_candidates: int = 100,
        include_vector: bool = False,
    ) -> List[Dict[str, Any]]:
        """knn-search news-<lang>. The read side of what add_article writes.

        Mirrors core/retrieval.py's ElasticRetriever so the crawler can verify
        its own output without standing up the API.
        """
        try:
            response = self.client.search(
                index=self.articles_index,
                knn={
                    "field": "content_vector",
                    "query_vector": self.embed_query(query),
                    "k": k,
                    "num_candidates": num_candidates,
                },
                source_excludes=None if include_vector else ["content_vector"],
            )
        except NotFoundError:
            return []
        except (ApiError, TransportError) as e:
            raise _wrap(f"POST /{self.articles_index}/_search", e) from e
        return [hit["_source"] for hit in response["hits"]["hits"]]


def from_env(lang: str) -> Optional[Indexer]:
    """Build an Indexer from ELASTIC_URL/API_KEY/INDEX/USER/PASSWORD, or None if unset.

    Elasticsearch is optional: with ELASTIC_URL unset the crawler behaves
    exactly as before and writes only the NoD corpus.

    ELASTIC_INDEX is the shared knob. core/config.py resolves its settings
    environment > .env > config.yaml, so the same variable overrides the API's
    `elastic_index` (config.yaml's default is news-german) and the name this
    crawler writes to -- which otherwise drift apart in silence, the API
    querying an index nothing fills. Unset, the derived news-<lang> stands.

    Values are stripped for the reason core/config.py strips its own: trailing
    whitespace in an .env line is invisible and produces an index name that
    looks identical in logs but does not match.
    """
    url = os.environ.get("ELASTIC_URL", "").strip()
    if not url:
        return None
    return Indexer(
        url=url,
        lang=lang,
        api_key=os.environ.get("ELASTIC_API_KEY", "").strip() or None,
        index=os.environ.get("ELASTIC_INDEX", "").strip() or None,
        #user=os.environ.get("ELASTIC_USER", "").strip() or None,
        #password=os.environ.get("ELASTIC_PASSWORD", "").strip() or None,
    )
