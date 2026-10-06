import re

import pytest

from app.agents.evidence import EvidenceExtractor
from app.agents.gap_analyzer import GapAnalyzer
from app.agents.planner import ResearchPlanner
from app.agents.writer import ReportWriter
from app.models.research import ResearchState, ResearchStatus
from app.services.llm import DemoLLMClient
from app.services.orchestrator import ResearchOrchestrator
from app.storage.repository import InMemoryResearchRepository
from app.tools.search import DemoSearchProvider
from app.tools.web_fetcher import PageFetcher


@pytest.mark.asyncio
async def test_full_orchestration_and_citation_mapping():
    llm = DemoLLMClient()
    repository = InMemoryResearchRepository()
    orchestrator = ResearchOrchestrator(
        repository, ResearchPlanner(llm), DemoSearchProvider(), PageFetcher(),
        EvidenceExtractor(llm), GapAnalyzer(llm), ReportWriter(llm),
        max_rounds=2, max_results=1, max_sources=4,
    )
    state = await repository.create(ResearchState(
        original_question="When does reranking most improve retrieval augmented generation systems?",
        demo_mode=True,
    ))
    await orchestrator.run(state)
    assert state.status == ResearchStatus.COMPLETED
    assert state.research_plan is not None
    assert state.sources and state.evidence
    citations = [int(value) for value in re.findall(r"\[(\d+)\]", state.final_report or "")]
    assert citations
    source_section = (state.final_report or "").split("## Sources", 1)[1]
    assert max(citations) <= len(re.findall(r"^\d+\.", source_section, re.MULTILINE))

