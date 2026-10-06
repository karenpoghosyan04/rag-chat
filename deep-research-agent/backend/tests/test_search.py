import pytest

from app.tools.search import DemoSearchProvider, SearchProvider


@pytest.mark.asyncio
async def test_demo_provider_obeys_search_contract():
    provider: SearchProvider = DemoSearchProvider()
    results = await provider.search("dense retrieval reranking", max_results=2)
    assert len(results) == 2
    assert all(result.provider == "demo" for result in results)
    assert all(result.url.startswith("https://") for result in results)

