import json
from abc import ABC, abstractmethod
from typing import TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel

from app.models.research import (
    Evidence,
    EvidenceBatch,
    GapAnalysis,
    ResearchPlan,
    SubQuestion,
    SubQuestionAssessment,
    WrittenReport,
)

T = TypeVar("T", bound=BaseModel)


class LLMClient(ABC):
    @abstractmethod
    async def structured_generate(
        self, *, system: str, prompt: str, response_model: type[T]
    ) -> T: ...

    @abstractmethod
    async def generate(self, *, system: str, prompt: str) -> str: ...


class OpenAILLMClient(LLMClient):
    def __init__(self, api_key: str, model: str) -> None:
        self.client = AsyncOpenAI(api_key=api_key, timeout=60, max_retries=2)
        self.model = model

    async def structured_generate(
        self, *, system: str, prompt: str, response_model: type[T]
    ) -> T:
        response = await self.client.responses.parse(
            model=self.model,
            instructions=system,
            input=prompt,
            text_format=response_model,
        )
        if response.output_parsed is None:
            raise ValueError("The model did not return valid structured output")
        return response.output_parsed

    async def generate(self, *, system: str, prompt: str) -> str:
        response = await self.client.responses.create(
            model=self.model, instructions=system, input=prompt
        )
        return response.output_text


class DemoLLMClient(LLMClient):
    """Deterministic offline model substitute; useful for setup, demos, and tests."""

    async def structured_generate(
        self, *, system: str, prompt: str, response_model: type[T]
    ) -> T:
        payload = json.loads(prompt)
        if response_model is ResearchPlan:
            q = payload["question"]
            topics = [
                ("foundations", f"What are the core approaches and how do they work for: {q}?"),
                ("evidence", "What empirical evidence compares their quality and efficiency?"),
                ("tradeoffs", "What are the main latency, cost, and implementation trade-offs?"),
                ("conditions", "Under which data and query conditions does each approach work best?"),
            ]
            return response_model(
                objective=f"Develop an evidence-based comparison that answers: {q}",
                subquestions=[
                    SubQuestion(
                        id=key,
                        question=text,
                        search_queries=[f"{q} {key} research", f"{q} {key} benchmark"],
                    )
                    for key, text in topics
                ],
            )
        if response_model is EvidenceBatch:
            source = payload["source"]
            subquestion = payload["subquestion"]
            text = source.get("text", "")
            snippet = " ".join(text.split())[:420]
            claim = source.get("summary") or snippet.split(". ")[0]
            return response_model(evidence=[Evidence(
                source_id=source["id"], source_title=source["title"],
                source_url=source["url"], subquestion=subquestion,
                claim=claim, supporting_text=snippet, relevance_score=0.78,
            )])
        if response_model is GapAnalysis:
            counts = payload["evidence_counts"]
            assessments = [SubQuestionAssessment(
                subquestion=item["question"], sufficient=counts.get(item["question"], 0) >= 1,
                rationale="At least one relevant source was collected." if counts.get(item["question"], 0) else "No relevant evidence was collected.",
            ) for item in payload["subquestions"]]
            missing = [a.subquestion for a in assessments if not a.sufficient]
            return response_model(
                assessments=assessments,
                missing_information=missing,
                additional_queries=[f"{m} systematic review" for m in missing][:4],
            )
        if response_model is WrittenReport:
            sources = payload["sources"]
            evidence = payload["evidence"]
            used = []
            bullets = []
            for item in evidence[:8]:
                sid = item["source_id"]
                if sid not in used:
                    used.append(sid)
                num = used.index(sid) + 1
                bullets.append(f"- {item['claim']} [{num}]")
            source_by_id = {s["id"]: s for s in sources}
            source_lines = [
                f"{i}. [{source_by_id[sid]['title']}]({source_by_id[sid]['url']})"
                for i, sid in enumerate(used, 1) if sid in source_by_id
            ]
            markdown = (
                "## Executive summary\n\n"
                "The available material supports a conditional, evidence-led answer rather than a universal winner. "
                "Retrieval quality, latency constraints, corpus characteristics, and evaluation design should determine the architecture.\n\n"
                "## Detailed findings\n\n" + "\n".join(bullets) +
                "\n\n## Trade-offs\n\n"
                "Higher-quality multi-stage retrieval generally adds latency and operational complexity. "
                "Measure gains on representative queries before adopting the most complex pipeline.\n\n"
                "## Uncertainties and limitations\n\n"
                "This report was produced in offline demo mode from a small embedded source set. "
                "Enable OpenAI and Tavily for question-specific web research and deeper synthesis.\n\n"
                "## Sources\n\n" + "\n".join(source_lines)
            )
            return response_model(markdown=markdown, cited_source_ids=used)
        raise TypeError(f"Demo client does not support {response_model.__name__}")

    async def generate(self, *, system: str, prompt: str) -> str:
        return "Demo mode does not perform free-form generation."

