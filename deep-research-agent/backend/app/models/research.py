from datetime import datetime, timezone
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field, HttpUrl


class ResearchStatus(StrEnum):
    QUEUED = "queued"
    PLANNING = "planning"
    SEARCHING = "searching"
    READING = "reading"
    EXTRACTING = "extracting"
    ANALYZING_GAPS = "analyzing_gaps"
    ADDITIONAL_RESEARCH = "additional_research"
    WRITING = "writing"
    COMPLETED = "completed"
    FAILED = "failed"


class SubQuestion(BaseModel):
    id: str
    question: str
    search_queries: list[str] = Field(min_length=1, max_length=3)


class ResearchPlan(BaseModel):
    objective: str
    subquestions: list[SubQuestion] = Field(min_length=3, max_length=6)


class SearchResult(BaseModel):
    query: str
    title: str
    url: str
    snippet: str = ""
    rank: int = 0
    provider: str
    content: str | None = Field(default=None, exclude=True)


class Source(BaseModel):
    id: str
    title: str
    url: str
    domain: str
    text: str = Field(exclude=True)
    fetch_status: str = "fetched"
    error: str | None = None
    queries: list[str] = Field(default_factory=list)


class Evidence(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    source_id: str
    source_title: str
    source_url: str
    subquestion: str
    claim: str
    supporting_text: str
    relevance_score: float = Field(ge=0, le=1)


class SubQuestionAssessment(BaseModel):
    subquestion: str
    sufficient: bool
    rationale: str


class GapAnalysis(BaseModel):
    assessments: list[SubQuestionAssessment]
    missing_information: list[str] = Field(default_factory=list)
    additional_queries: list[str] = Field(default_factory=list, max_length=4)


class ResearchEvent(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: ResearchStatus
    message: str


class ResearchState(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid4()))
    original_question: str
    research_plan: ResearchPlan | None = None
    search_queries: list[str] = Field(default_factory=list)
    search_results: list[SearchResult] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    completed_subquestions: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    current_round: int = 0
    status: ResearchStatus = ResearchStatus.QUEUED
    final_report: str | None = None
    error: str | None = None
    demo_mode: bool = False
    events: list[ResearchEvent] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CreateResearchRequest(BaseModel):
    question: str = Field(min_length=10, max_length=3000)


class CreateResearchResponse(BaseModel):
    task_id: str
    status: ResearchStatus


class EvidenceBatch(BaseModel):
    evidence: list[Evidence] = Field(default_factory=list)


class WrittenReport(BaseModel):
    markdown: str
    cited_source_ids: list[str]

