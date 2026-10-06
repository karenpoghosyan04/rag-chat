import json

from app.models.research import Evidence, EvidenceBatch, Source, SubQuestion
from app.services.llm import LLMClient


class EvidenceExtractor:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def extract(self, source: Source, subquestion: SubQuestion) -> list[Evidence]:
        if not source.text.strip():
            return []
        result = await self.llm.structured_generate(
            system=("Extract only evidence directly supported by the source for the subquestion. "
                    "Return at most 3 concise claims with short supporting excerpts. Never invent support."),
            prompt=json.dumps({
                "subquestion": subquestion.question,
                "source": {"id": source.id, "title": source.title, "url": source.url,
                           "text": source.text[:12000], "summary": source.text[:300]},
            }), response_model=EvidenceBatch,
        )
        return [item for item in result.evidence if item.relevance_score >= 0.45]

