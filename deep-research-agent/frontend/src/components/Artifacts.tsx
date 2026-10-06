import { ExternalLink, Search, ShieldCheck } from 'lucide-react'
import type { ResearchState } from '../types/research'

export function Artifacts({ state }: { state: ResearchState }) {
  return <div className="space-y-5">
    {state.research_plan && <section className="panel">
      <div className="eyebrow">Research plan</div>
      <p className="mt-2 text-sm font-medium leading-6 text-ink">{state.research_plan.objective}</p>
      <ol className="mt-4 space-y-3">
        {state.research_plan.subquestions.map((sq, index) => <li key={sq.id} className="flex gap-3 text-sm leading-5 text-slate-600">
          <span className="number">{index + 1}</span><span>{sq.question}</span>
        </li>)}
      </ol>
    </section>}

    {state.search_queries.length > 0 && <section className="panel">
      <div className="eyebrow">Queries · {state.search_queries.length}</div>
      <div className="mt-3 flex flex-wrap gap-2">{state.search_queries.map((q, i) => <span key={`${q}-${i}`} className="chip"><Search className="h-3 w-3" />{q}</span>)}</div>
    </section>}

    {state.sources.length > 0 && <section className="panel">
      <div className="eyebrow">Sources inspected · {state.sources.length}</div>
      <div className="mt-2 divide-y divide-line">{state.sources.map(source => <a key={source.id} href={source.url} target="_blank" rel="noreferrer" className="group flex items-start justify-between gap-3 py-3">
        <span><span className="block text-sm font-medium leading-5 text-ink group-hover:text-moss">{source.title}</span><span className="mt-1 block text-xs text-slate-400">{source.domain}</span></span>
        <span className={`mt-0.5 rounded-full px-2 py-1 text-[11px] font-semibold ${source.fetch_status === 'fetched' ? 'bg-emerald-50 text-emerald-700' : 'bg-amber-50 text-amber-700'}`}>{source.fetch_status}</span>
      </a>)}</div>
    </section>}

    {state.evidence.length > 0 && <section>
      <div className="mb-3 flex items-center justify-between"><div className="eyebrow">Evidence collected</div><span className="text-xs text-slate-400">{state.evidence.length} claims</span></div>
      <div className="grid gap-3">{state.evidence.slice(0, 10).map(item => <article key={item.id} className="panel">
        <div className="flex items-center gap-2 text-xs font-semibold text-moss"><ShieldCheck className="h-3.5 w-3.5" />{Math.round(item.relevance_score * 100)}% relevance</div>
        <p className="mt-2 text-sm font-medium leading-6 text-ink">{item.claim}</p>
        <a href={item.source_url} target="_blank" rel="noreferrer" className="mt-3 flex items-center gap-1 text-xs text-slate-400 hover:text-moss">{item.source_title}<ExternalLink className="h-3 w-3" /></a>
      </article>)}</div>
    </section>}
  </div>
}

