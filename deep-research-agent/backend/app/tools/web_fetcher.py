from urllib.parse import urlparse

import httpx
import trafilatura
from bs4 import BeautifulSoup

from app.models.research import SearchResult, Source


class PageFetcher:
    def __init__(self, timeout: float = 20, max_chars: int = 18_000) -> None:
        self.timeout = timeout
        self.max_chars = max_chars

    async def fetch(self, result: SearchResult, source_id: str) -> Source:
        domain = urlparse(result.url).netloc.removeprefix("www.")
        if result.content:
            return Source(
                id=source_id, title=result.title, url=result.url, domain=domain,
                text=result.content[: self.max_chars], queries=[result.query],
            )
        try:
            headers = {"User-Agent": "DeepResearchPrototype/0.1 (+local research tool)"}
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True, headers=headers) as client:
                response = await client.get(result.url)
                response.raise_for_status()
            extracted = trafilatura.extract(response.text, include_links=False, include_tables=True)
            if not extracted:
                soup = BeautifulSoup(response.text, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "aside"]):
                    tag.decompose()
                extracted = soup.get_text(" ", strip=True)
            if not extracted:
                raise ValueError("No readable text found")
            title = BeautifulSoup(response.text, "html.parser").title
            return Source(
                id=source_id, title=title.get_text(strip=True) if title else result.title,
                url=str(response.url), domain=urlparse(str(response.url)).netloc.removeprefix("www."),
                text=extracted[: self.max_chars], queries=[result.query],
            )
        except Exception as exc:
            return Source(
                id=source_id, title=result.title, url=result.url, domain=domain,
                text=result.snippet, fetch_status="failed", error=str(exc)[:240], queries=[result.query],
            )

