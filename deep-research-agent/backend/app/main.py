import asyncio
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.agents.evidence import EvidenceExtractor
from app.agents.gap_analyzer import GapAnalyzer
from app.agents.planner import ResearchPlanner
from app.agents.writer import ReportWriter
from app.config import get_settings
from app.models.research import CreateResearchRequest, CreateResearchResponse, ResearchState
from app.services.llm import DemoLLMClient, OpenAILLMClient
from app.services.orchestrator import ResearchOrchestrator
from app.storage.repository import InMemoryResearchRepository
from app.tools.search import DemoSearchProvider, TavilySearchProvider
from app.tools.web_fetcher import PageFetcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
settings = get_settings()
repository = InMemoryResearchRepository()
llm = OpenAILLMClient(settings.openai_api_key, settings.llm_model) if settings.openai_api_key else DemoLLMClient()
search = (TavilySearchProvider(settings.tavily_api_key)
          if settings.search_provider.lower() == "tavily" and settings.tavily_api_key
          else DemoSearchProvider())
orchestrator = ResearchOrchestrator(
    repository, ResearchPlanner(llm), search, PageFetcher(), EvidenceExtractor(llm),
    GapAnalyzer(llm), ReportWriter(llm), max_rounds=settings.max_research_rounds,
    max_results=settings.max_results_per_query, max_sources=settings.max_sources,
)

app = FastAPI(title="Deep Research Agent", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[f"http://localhost:{settings.frontend_port}", f"http://127.0.0.1:{settings.frontend_port}"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> dict[str, str | bool]:
    return {"status": "ok", "demo_mode": isinstance(llm, DemoLLMClient) or isinstance(search, DemoSearchProvider)}


@app.post("/api/research", response_model=CreateResearchResponse, status_code=202)
async def create_research(request: CreateResearchRequest) -> CreateResearchResponse:
    state = ResearchState(
        original_question=request.question.strip(),
        demo_mode=isinstance(llm, DemoLLMClient) or isinstance(search, DemoSearchProvider),
    )
    await repository.create(state)
    logging.getLogger(__name__).info("task=%s created", state.task_id)
    asyncio.create_task(orchestrator.run(state))
    return CreateResearchResponse(task_id=state.task_id, status=state.status)


@app.get("/api/research/{task_id}", response_model=ResearchState)
async def get_research(task_id: str) -> ResearchState:
    state = await repository.get(task_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Research task not found")
    return state

