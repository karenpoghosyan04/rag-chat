import asyncio
import logging
from datetime import datetime, timezone

from app.agents.evidence import EvidenceExtractor
from app.agents.gap_analyzer import GapAnalyzer
from app.agents.planner import ResearchPlanner
from app.agents.writer import ReportWriter
from app.models.research import ResearchEvent, ResearchState, ResearchStatus, SubQuestion
from app.storage.repository import ResearchRepository
from app.tools.search import SearchProvider
from app.tools.web_fetcher import PageFetcher

logger = logging.getLogger(__name__)


class ResearchOrchestrator:
    def __init__(self, repository: ResearchRepository, planner: ResearchPlanner,
                 search: SearchProvider, fetcher: PageFetcher, extractor: EvidenceExtractor,
                 gap_analyzer: GapAnalyzer, writer: ReportWriter, *, max_rounds: int = 2,
                 max_results: int = 3, max_sources: int = 12) -> None:
        self.repository = repository
        self.planner = planner
        self.search = search
        self.fetcher = fetcher
        self.extractor = extractor
        self.gap_analyzer = gap_analyzer
        self.writer = writer
        self.max_rounds = max_rounds
        self.max_results = max_results
        self.max_sources = max_sources

    async def _stage(self, state: ResearchState, status: ResearchStatus, message: str) -> None:
        state.status = status
        state.updated_at = datetime.now(timezone.utc)
        state.events.append(ResearchEvent(status=status, message=message))
        await self.repository.save(state)
        logger.info("task=%s status=%s %s", state.task_id, status, message)

    async def run(self, state: ResearchState) -> None:
        try:
            await self._stage(state, ResearchStatus.PLANNING, "Planning the research approach")
            state.research_plan = await self.planner.plan(state.original_question)
            queries = [q for sq in state.research_plan.subquestions for q in sq.search_queries]

            for round_number in range(1, self.max_rounds + 1):
                state.current_round = round_number
                status = ResearchStatus.SEARCHING if round_number == 1 else ResearchStatus.ADDITIONAL_RESEARCH
                await self._stage(state, status, f"Running {len(queries)} searches in round {round_number}")
                new_results = []
                search_batches = await asyncio.gather(
                    *(self.search.search(query, self.max_results) for query in queries),
                    return_exceptions=True,
                )
                for query, batch in zip(queries, search_batches):
                    state.search_queries.append(query)
                    if isinstance(batch, Exception):
                        logger.warning("task=%s search_failed query=%r error=%s", state.task_id, query, batch)
                    else:
                        new_results.extend(batch)
                seen_urls = {item.url for item in state.search_results}
                new_results = [item for item in new_results if item.url not in seen_urls]
                state.search_results.extend(new_results)
                logger.info("task=%s round=%s results=%s", state.task_id, round_number, len(new_results))

                remaining = max(0, self.max_sources - len(state.sources))
                await self._stage(state, ResearchStatus.READING, f"Reading up to {remaining} new sources")
                fetches = await asyncio.gather(*(
                    self.fetcher.fetch(result, f"S{len(state.sources) + index}")
                    for index, result in enumerate(new_results[:remaining], 1)
                ))
                state.sources.extend(fetches)
                logger.info("task=%s fetched=%s successful=%s", state.task_id, len(fetches), sum(s.fetch_status == "fetched" for s in fetches))

                await self._stage(state, ResearchStatus.EXTRACTING, "Extracting question-specific evidence")
                extraction_jobs = []
                for source in fetches:
                    for subquestion in state.research_plan.subquestions:
                        extraction_jobs.append(self.extractor.extract(source, subquestion))
                batches = await asyncio.gather(*extraction_jobs, return_exceptions=True)
                for batch in batches:
                    if not isinstance(batch, Exception):
                        state.evidence.extend(batch)
                logger.info("task=%s evidence=%s", state.task_id, len(state.evidence))

                await self._stage(state, ResearchStatus.ANALYZING_GAPS, "Checking coverage and identifying gaps")
                gaps = await self.gap_analyzer.analyze(state.research_plan, state.evidence)
                state.completed_subquestions = [a.subquestion for a in gaps.assessments if a.sufficient]
                state.missing_information = gaps.missing_information
                if not gaps.additional_queries or round_number >= self.max_rounds:
                    break
                queries = gaps.additional_queries

            await self._stage(state, ResearchStatus.WRITING, "Synthesizing the cited report")
            state.final_report = await self.writer.write(
                state.original_question, state.research_plan, state.sources, state.evidence
            )
            await self._stage(state, ResearchStatus.COMPLETED, "Research report complete")
            logger.info("task=%s report_generated", state.task_id)
        except Exception as exc:
            logger.exception("task=%s failed", state.task_id)
            state.error = str(exc)
            await self._stage(state, ResearchStatus.FAILED, "Research failed; see the error for details")

