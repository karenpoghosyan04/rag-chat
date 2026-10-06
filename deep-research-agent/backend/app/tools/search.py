from abc import ABC, abstractmethod

import httpx

from app.models.research import SearchResult


class SearchProvider(ABC):
    name: str

    @abstractmethod
    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]: ...


class TavilySearchProvider(SearchProvider):
    name = "tavily"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        async with httpx.AsyncClient(timeout=25) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={"api_key": self.api_key, "query": query, "max_results": max_results, "search_depth": "advanced"},
            )
            response.raise_for_status()
        return [SearchResult(
            query=query, title=item.get("title") or item["url"], url=item["url"],
            snippet=item.get("content", ""), rank=index, provider=self.name,
        ) for index, item in enumerate(response.json().get("results", []), 1)]


class DemoSearchProvider(SearchProvider):
    name = "demo"
    _documents = [
        {
            "title": "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction",
            "url": "https://arxiv.org/abs/2004.12832",
            "text": "ColBERT uses contextualized token-level representations and late interaction. It retains fine-grained matching while precomputing document representations, offering a quality-efficiency trade-off between single-vector dense retrieval and full cross-encoder scoring.",
        },
        {
            "title": "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models",
            "url": "https://arxiv.org/abs/2104.08663",
            "text": "BEIR evaluates lexical and dense retrieval across diverse datasets. BM25 is a robust baseline and dense models do not consistently dominate out of domain. Reranking and late-interaction approaches can improve effectiveness, with performance varying by dataset.",
        },
        {
            "title": "Precise Zero-Shot Dense Retrieval without Relevance Labels",
            "url": "https://arxiv.org/abs/2212.10496",
            "text": "HyDE generates a hypothetical document and encodes it for dense retrieval. The results show that query transformation can improve zero-shot retrieval, highlighting how representation and domain shift affect first-stage recall.",
        },
        {
            "title": "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
            "url": "https://arxiv.org/abs/1908.10084",
            "text": "Bi-encoder sentence embeddings make semantic similarity search efficient by encoding candidates independently. Cross-encoders are typically more accurate but require pairwise inference, making them expensive for large candidate sets and suitable for reranking a shortlist.",
        },
        {
            "title": "MS MARCO: A Human Generated Machine Reading Comprehension Dataset",
            "url": "https://arxiv.org/abs/1611.09268",
            "text": "MS MARCO provides large-scale real query and passage data used to train and evaluate passage ranking systems. Its ranking tasks helped establish multi-stage retrieval evaluation using metrics such as MRR.",
        },
        {
            "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
            "url": "https://arxiv.org/abs/2005.11401",
            "text": "RAG combines parametric generation with retrieved non-parametric memory. Retrieval provides provenance and enables knowledge updates, while generation quality depends on retrieving relevant evidence into a limited context.",
        },
    ]

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        terms = set(query.lower().split())
        ranked = sorted(
            self._documents,
            key=lambda d: sum(term in (d["title"] + " " + d["text"]).lower() for term in terms),
            reverse=True,
        )[:max_results]
        return [SearchResult(
            query=query, title=item["title"], url=item["url"],
            snippet=item["text"][:240], content=item["text"], rank=index, provider=self.name,
        ) for index, item in enumerate(ranked, 1)]

