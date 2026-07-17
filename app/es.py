"""Elasticsearch indexing for the news crawler.

ES Index  for the new  crawler 

    news-<lang>/article      {title, content, url, published, guid}
    failures-<lang>/failure  {message, url, status, time}

Keyed by uid


Talks to Elasticsearch over the REST API with urllib rather than the
elasticsearch client library: the NoD stack pins Elasticsearch 5.5.

Mapping is left to Elasticsearch's dynamic detection,
"""

from __future__ import annotations

import base64
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ARTICLE_TYPE = "article"
FAILURE_TYPE = "failure"

# Elasticsearch rejects oversized _bulk bodies; chunk well below the default
# 100MB http.max_content_length since article bodies are unbounded.
_BULK_CHUNK = 200


class ElasticError(RuntimeError):
    """Raised when Elasticsearch is configured but a request fails.

    `status` carries the HTTP code so callers can tell "index does not exist
    yet" (404, expected on a first run) from a real fault they must not
    silently ignore.
    """

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


class Indexer:
    """Buffered writer for the news-<lang> / failures-<lang> indices."""

    def __init__(
        self,
        url: str,
        lang: str,
        user: Optional[str] = None,
        password: Optional[str] = None,
        timeout_s: int = 30,
    ):
        self.url = url.rstrip("/")
        self.lang = lang
        self.timeout_s = timeout_s
        self.articles_index = f"news-{lang}"
        self.failures_index = f"failures-{lang}"
        self.indexed = 0
        self.failures_logged = 0
        self._auth: Optional[str] = None
        if user and password:
            token = base64.b64encode(f"{user}:{password}".encode()).decode()
            self._auth = f"Basic {token}"
        self._bulk_lines: List[str] = []

    # -- transport ---------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[str] = None,
        content_type: str = "application/json",
    ) -> Dict[str, Any]:
        headers = {"Content-Type": content_type}
        if self._auth:
            headers["Authorization"] = self._auth
        data = body.encode("utf-8") if body is not None else None
        req = Request(self.url + path, data=data, headers=headers, method=method)
        try:
            with urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8")
        except HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:500]
            raise ElasticError(f"{method} {path} -> HTTP {e.code}: {detail}", status=e.code) from e
        except URLError as e:
            raise ElasticError(f"{method} {path} -> {e.reason}") from e
        return json.loads(raw) if raw else {}

    def ping(self) -> str:
        """Return the Elasticsearch version, raising ElasticError if unreachable."""
        info = self._request("GET", "/")
        return str(info.get("version", {}).get("number", "unknown"))

    # -- writing -----------------------------------------------------------

    def add_article(
        self,
        title: str,
        content: str,
        url: str,
        published: datetime,
        guid: str,
    ) -> None:
        """Queue an article. Matches the Go FeedItem field names."""
        doc_id = guid or url
        action = {"index": {"_index": self.articles_index, "_type": ARTICLE_TYPE, "_id": doc_id}}
        doc = {
            "title": title,
            "content": content,
            "url": url,
            "published": published.isoformat(),
            "guid": doc_id,
        }
        self._bulk_lines.append(json.dumps(action))
        self._bulk_lines.append(json.dumps(doc))
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
        action = {"index": {"_index": self.failures_index, "_type": FAILURE_TYPE, "_id": doc_id}}
        doc = {
            "message": message,
            "url": url,
            "status": status,
            "time": time.isoformat(),
        }
        self._bulk_lines.append(json.dumps(action))
        self._bulk_lines.append(json.dumps(doc))
        self.failures_logged += 1
        self._maybe_send()

    def _maybe_send(self) -> None:
        if len(self._bulk_lines) >= _BULK_CHUNK * 2:
            self._send_bulk()

    def _send_bulk(self) -> None:
        """Send buffered documents. Raises ElasticError if any item was rejected."""
        if not self._bulk_lines:
            return
        body = "\n".join(self._bulk_lines) + "\n"
        self._bulk_lines = []
        result = self._request("POST", "/_bulk", body, content_type="application/x-ndjson")
        if result.get("errors"):
            reasons = []
            for item in result.get("items", []):
                info = next(iter(item.values()), {})
                if info.get("error"):
                    reasons.append(str(info["error"].get("reason", info["error"])))
            raise ElasticError(f"{len(reasons)} document(s) rejected: {'; '.join(reasons[:3])}")

    def refresh(self) -> None:
        """Make freshly indexed documents searchable."""
        for index in (self.articles_index, self.failures_index):
            try:
                self._request("POST", f"/{index}/_refresh")
            except ElasticError as e:
                # A refresh on an index this run never wrote to is not a fault.
                if e.status != 404:
                    raise

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
            body = json.dumps({"ids": chunk})
            try:
                result = self._request(
                    "POST", f"/{self.articles_index}/{ARTICLE_TYPE}/_mget?_source=false", body
                )
            except ElasticError as e:
                if e.status == 404:
                    return found
                raise
            for doc in result.get("docs", []):
                if doc.get("found"):
                    found.add(doc["_id"])
        return found

    def articles_for_day(self, day: datetime, time_zone: str) -> List[Dict[str, Any]]:
        """Fetch every article published on `day`, interpreted in `time_zone`.

        Elasticsearch rounds a bare date down for gte and up for lte, so a single day string covers the
        whole day. 

        Only a missing index yields an empty result; other errors are raised
        so a transient outage cannot quietly produce an empty corpus that then
        overwrites a good one.
        """
        day_str = day.strftime("%Y-%m-%d")
        query = {
            "size": 1000,
            "query": {
                "range": {
                    "published": {
                        "gte": day_str,
                        "lte": day_str,
                        "time_zone": time_zone,
                        "format": "yyyy-MM-dd",
                    }
                }
            },
        }
        articles: List[Dict[str, Any]] = []
        try:
            result = self._request(
                "POST",
                f"/{self.articles_index}/{ARTICLE_TYPE}/_search?scroll=2m",
                json.dumps(query),
            )
        except ElasticError as e:
            if e.status == 404:
                return articles
            raise

        scroll_id = None
        try:
            while True:
                hits = result.get("hits", {}).get("hits", [])
                if not hits:
                    break
                articles.extend(h["_source"] for h in hits)
                scroll_id = result.get("_scroll_id")
                if not scroll_id:
                    break
                result = self._request(
                    "POST",
                    "/_search/scroll",
                    json.dumps({"scroll": "2m", "scroll_id": scroll_id}),
                )
        finally:
            # Scroll contexts hold Lucene segments open until they expire.
            if scroll_id:
                try:
                    self._request(
                        "DELETE", "/_search/scroll", json.dumps({"scroll_id": [scroll_id]})
                    )
                except ElasticError:
                    pass
        return articles


def from_env(lang: str) -> Optional[Indexer]:
    """Build an Indexer from ELASTIC_URL/USER/PASSWORD, or None if unset.

    Elasticsearch is optional: with ELASTIC_URL unset the crawler behaves
    exactly as before and writes only the NoD corpus.
    """
    url = os.environ.get("ELASTIC_URL", "").strip()
    if not url:
        return None
    return Indexer(
        url=url,
        lang=lang,
        user=os.environ.get("ELASTIC_USER", "").strip() or None,
        password=os.environ.get("ELASTIC_PASSWORD", "").strip() or None,
    )
