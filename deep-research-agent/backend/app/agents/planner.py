import json

from app.models.research import ResearchPlan
from app.services.llm import LLMClient


class ResearchPlanner:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def plan(self, question: str) -> ResearchPlan:
        return await self.llm.structured_generate(
            system=("You are a research planner. Clarify the objective and create 3-6 distinct, answerable "
                    "subquestions with 1-3 precise web queries each. Do not answer the question."),
            prompt=json.dumps({"question": question}), response_model=ResearchPlan,
        )

