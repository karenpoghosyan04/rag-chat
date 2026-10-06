import json
from collections import Counter

from app.models.research import Evidence, GapAnalysis, ResearchPlan
from app.services.llm import LLMClient


class GapAnalyzer:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def analyze(self, plan: ResearchPlan, evidence: list[Evidence]) -> GapAnalysis:
        counts = Counter(item.subquestion for item in evidence)
        return await self.llm.structured_generate(
            system=("Assess whether each subquestion has enough diverse, relevant evidence. Identify concrete gaps "
                    "and propose no more than four targeted follow-up searches. Be conservative."),
            prompt=json.dumps({
                "subquestions": [sq.model_dump() for sq in plan.subquestions],
                "evidence_counts": dict(counts),
                "evidence": [e.model_dump() for e in evidence],
            }), response_model=GapAnalysis,
        )

