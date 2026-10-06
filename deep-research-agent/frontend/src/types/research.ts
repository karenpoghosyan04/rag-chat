export type Status = 'queued' | 'planning' | 'searching' | 'reading' | 'extracting' |
  'analyzing_gaps' | 'additional_research' | 'writing' | 'completed' | 'failed'

export interface SubQuestion { id: string; question: string; search_queries: string[] }
export interface ResearchPlan { objective: string; subquestions: SubQuestion[] }
export interface SearchResult { query: string; title: string; url: string; snippet: string; rank: number; provider: string }
export interface Source { id: string; title: string; url: string; domain: string; fetch_status: string; error?: string; queries: string[] }
export interface Evidence { id: string; source_id: string; source_title: string; source_url: string; subquestion: string; claim: string; supporting_text: string; relevance_score: number }
export interface ResearchEvent { timestamp: string; status: Status; message: string }
export interface ResearchState {
  task_id: string; original_question: string; research_plan?: ResearchPlan;
  search_queries: string[]; search_results: SearchResult[]; sources: Source[];
  evidence: Evidence[]; completed_subquestions: string[]; missing_information: string[];
  current_round: number; status: Status; final_report?: string; error?: string;
  demo_mode: boolean; events: ResearchEvent[]; created_at: string; updated_at: string;
}

