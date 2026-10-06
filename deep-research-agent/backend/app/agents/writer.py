import json
import re

from app.models.research import Evidence, ResearchPlan, Source, WrittenReport
from app.services.llm import LLMClient


class ReportWriter:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def write(self, question: str, plan: ResearchPlan, sources: list[Source], evidence: list[Evidence]) -> str:
        result = await self.llm.structured_generate(
            system=("Write a rigorous Markdown research report with: Executive summary, Detailed findings, "
                    "Trade-offs, Uncertainties and limitations, and Sources. Use only supplied evidence. "
                    "Cite claims inline as [1], [2], etc. Number sources in first-citation order and ensure every "
                    "citation maps to a supplied source. Do not cite unused sources."),
            prompt=json.dumps({
                "question": question, "plan": plan.model_dump(),
                "sources": [{"id": s.id, "title": s.title, "url": s.url, "domain": s.domain} for s in sources],
                "evidence": [e.model_dump() for e in evidence],
            }), response_model=WrittenReport,
        )
        valid_ids = {source.id for source in sources}
        if any(source_id not in valid_ids for source_id in result.cited_source_ids):
            raise ValueError("Report referenced an unknown source")
        citations = {int(n) for n in re.findall(r"\[(\d+)\]", result.markdown)}
        expected = set(range(1, len(result.cited_source_ids) + 1))
        if citations and not citations.issubset(expected):
            raise ValueError("Report contains a citation without a source mapping")
        return result.markdown

