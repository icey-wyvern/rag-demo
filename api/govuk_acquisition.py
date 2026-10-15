"""GOV.UK Content API acquisition and normalization boundary."""

from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
from urllib.parse import urlsplit
from uuid import UUID

import httpx

from .versioned_ingest import ObservedSourceVersion


API_ORIGIN = "https://www.gov.uk"
USER_AGENT = "rag-demo-versioned-ingest/0.1 (+https://github.com/icey-wyvern/rag-demo)"


class GovUkAcquisitionError(ValueError):
    """A source cannot be acquired or normalized safely."""


def _base_path(value: object) -> str:
    if not isinstance(value, str) or not value.startswith("/") or value.startswith("//"):
        raise GovUkAcquisitionError("GOV.UK base path must start with one slash")
    parsed = urlsplit(value)
    if parsed.query or parsed.fragment or parsed.scheme or parsed.netloc or ".." in parsed.path.split("/"):
        raise GovUkAcquisitionError("GOV.UK base path contains an unsupported component")
    return value


@dataclass(frozen=True)
class AcquiredGovUkContent:
    base_path: str
    request_url: str
    payload: dict
    observed_at: datetime


class GovUkClient:
    def __init__(
        self,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout_seconds: float = 10.0,
        max_attempts: int = 3,
        max_response_bytes: int = 2_000_000,
        clock=lambda: datetime.now(timezone.utc),
    ):
        if timeout_seconds <= 0 or not 1 <= max_attempts <= 5 or max_response_bytes <= 0:
            raise ValueError("Invalid GOV.UK HTTP limits")
        self.transport = transport
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.max_response_bytes = max_response_bytes
        self.clock = clock

    def fetch(self, base_path: str) -> AcquiredGovUkContent:
        path = _base_path(base_path)
        url = f"{API_ORIGIN}/api/content{path}"
        with httpx.Client(
            transport=self.transport,
            timeout=self.timeout_seconds,
            follow_redirects=False,
            headers={"Accept": "application/json", "User-Agent": USER_AGENT},
        ) as client:
            for attempt in range(self.max_attempts):
                try:
                    with client.stream("GET", url) as response:
                        if response.status_code in (429, 502, 503, 504) and attempt + 1 < self.max_attempts:
                            continue
                        if response.status_code != 200:
                            raise GovUkAcquisitionError(
                                f"GOV.UK Content API returned HTTP {response.status_code} for {path}"
                            )
                        size = 0
                        parts = []
                        for part in response.iter_bytes():
                            size += len(part)
                            if size > self.max_response_bytes:
                                raise GovUkAcquisitionError(f"GOV.UK response exceeds size limit for {path}")
                            parts.append(part)
                    try:
                        payload = json.loads(b"".join(parts))
                    except (UnicodeError, json.JSONDecodeError) as exc:
                        raise GovUkAcquisitionError(f"GOV.UK returned malformed JSON for {path}") from exc
                    if not isinstance(payload, dict):
                        raise GovUkAcquisitionError(f"GOV.UK returned a non-object response for {path}")
                    return AcquiredGovUkContent(path, url, payload, self.clock())
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    if attempt + 1 == self.max_attempts:
                        raise GovUkAcquisitionError(f"GOV.UK request failed for {path}: {type(exc).__name__}") from exc
        raise AssertionError("retry loop exhausted")


class _GuidanceHtml(HTMLParser):
    BLOCKS = {"h1", "h2", "h3", "h4", "p", "li", "blockquote"}
    SKIP = {"script", "style", "nav", "aside", "footer"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.current: list[str] = []
        self.active: str | None = None
        self.skipping = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.SKIP:
            self.skipping += 1
        elif not self.skipping and tag in self.BLOCKS:
            self._flush()
            self.active = tag

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP and self.skipping:
            self.skipping -= 1
        elif not self.skipping and tag == self.active:
            self._flush()
            self.active = None

    def handle_data(self, data: str) -> None:
        if self.active and not self.skipping:
            self.current.append(data)

    def _flush(self) -> None:
        value = " ".join(" ".join(self.current).split())
        if value:
            self.parts.append(value)
        self.current = []

    def text(self) -> str:
        self._flush()
        return "\n\n".join(self.parts)


def _timestamp(payload: dict, name: str) -> datetime | None:
    raw = payload.get(name)
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise GovUkAcquisitionError(f"Invalid GOV.UK {name} timestamp")
    try:
        value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise GovUkAcquisitionError(f"Invalid GOV.UK {name} timestamp") from exc
    if value.tzinfo is None or value.utcoffset() is None:
        raise GovUkAcquisitionError(f"GOV.UK {name} timestamp has no timezone")
    return value


@dataclass(frozen=True)
class NormalizedGovUkContent:
    content_id: str
    base_path: str
    schema_name: str
    document_type: str
    observed: ObservedSourceVersion


class GovUkNormalizer:
    def normalize(self, acquired: AcquiredGovUkContent) -> NormalizedGovUkContent:
        payload = acquired.payload
        path = _base_path(payload.get("base_path"))
        if path != acquired.base_path:
            raise GovUkAcquisitionError("GOV.UK response base path differs from requested path")
        raw_id = payload.get("content_id")
        try:
            content_id = str(UUID(raw_id))
        except (ValueError, TypeError, AttributeError) as exc:
            raise GovUkAcquisitionError("GOV.UK content_id must be a UUID") from exc
        title = payload.get("title")
        schema_name = payload.get("schema_name")
        document_type = payload.get("document_type")
        if not isinstance(title, str) or not title.strip():
            raise GovUkAcquisitionError("GOV.UK response has no title")
        if schema_name != "service_manual_guide" or document_type != "service_manual_guide":
            raise GovUkAcquisitionError(f"Unsupported GOV.UK content schema for {path}")
        details = payload.get("details")
        body = details.get("body") if isinstance(details, dict) else None
        if not isinstance(body, str) or not body.strip():
            raise GovUkAcquisitionError(f"GOV.UK guidance body is missing for {path}")
        parser = _GuidanceHtml()
        parser.feed(body)
        content = parser.text()
        if not content:
            raise GovUkAcquisitionError(f"GOV.UK guidance body has no usable text for {path}")
        observed = ObservedSourceVersion(
            source_key=f"govuk:{content_id}",
            canonical_url=f"{API_ORIGIN}{path}",
            title=title.strip(),
            retrieved_url=acquired.request_url,
            content=content.encode("utf-8"),
            observed_at=acquired.observed_at,
            source_published_at=_timestamp(payload, "first_published_at"),
            source_updated_at=_timestamp(payload, "public_updated_at"),
        )
        return NormalizedGovUkContent(content_id, path, schema_name, document_type, observed)
