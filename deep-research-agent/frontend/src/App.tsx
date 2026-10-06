import { FormEvent, useEffect, useState } from 'react'
import { ArrowUpRight, BookOpenText, FlaskConical, Sparkles } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import { Artifacts } from './components/Artifacts'
import { StatusRail } from './components/StatusRail'
import { createResearch, getResearch } from './services/api'
import type { ResearchState } from './types/research'

const example = 'Compare dense retrieval, hybrid retrieval, and cross-encoder reranking for modern RAG systems. When does reranking provide the largest benefit?'

export default function App() {
  const [question, setQuestion] = useState(example)
  const [taskId, setTaskId] = useState<string>()
  const [state, setState] = useState<ResearchState>()
  const [error, setError] = useState<string>()
  const busy = !!taskId && state?.status !== 'completed' && state?.status !== 'failed'

  useEffect(() => {
    if (!taskId || !busy) return
    const poll = async () => {
      try { setState(await getResearch(taskId)) } catch (e) { setError(e instanceof Error ? e.message : 'Connection lost') }
    }
    poll()
    const timer = window.setInterval(poll, 900)
    return () => window.clearInterval(timer)
  }, [taskId, busy])

  async function submit(event: FormEvent) {
    event.preventDefault(); setError(undefined); setState(undefined)
    try { setTaskId(await createResearch(question)) } catch (e) { setError(e instanceof Error ? e.message : 'Could not start') }
  }

  return <main className="min-h-screen">
    <header className="border-b border-line bg-white/80 backdrop-blur">
      <div className="mx-auto flex max-w-[1500px] items-center justify-between px-5 py-4 lg:px-8">
        <div className="flex items-center gap-3"><div className="grid h-9 w-9 place-items-center rounded-xl bg-ink text-white"><BookOpenText className="h-4 w-4" /></div><div><div className="font-semibold tracking-tight text-ink">Fieldnote</div><div className="text-xs text-slate-400">Deep research workspace</div></div></div>
        <div className="flex items-center gap-2 rounded-full border border-line bg-paper px-3 py-1.5 text-xs font-medium text-slate-500"><FlaskConical className="h-3.5 w-3.5" />Local prototype</div>
      </div>
    </header>

    <div className="mx-auto max-w-[1500px] px-5 py-8 lg:px-8">
      <section className="mb-8 max-w-4xl">
        <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-moss"><Sparkles className="h-4 w-4" />Start with a consequential question</div>
        <form onSubmit={submit} className="rounded-2xl border border-line bg-white p-2 shadow-soft">
          <textarea aria-label="Research question" value={question} onChange={e => setQuestion(e.target.value)} rows={3} className="w-full resize-none rounded-xl border-0 bg-transparent px-4 py-3 text-lg leading-8 text-ink outline-none placeholder:text-slate-300" placeholder="Investigate when hybrid retrieval outperforms dense retrieval in RAG systems and summarize the evidence." />
          <div className="flex items-center justify-between border-t border-line px-3 pt-2">
            <span className="text-xs text-slate-400">Up to 2 research rounds · evidence stays inspectable</span>
            <button disabled={busy || question.trim().length < 10} className="flex items-center gap-2 rounded-xl bg-ink px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-moss disabled:cursor-not-allowed disabled:opacity-40">{busy ? 'Researching…' : 'Start research'}<ArrowUpRight className="h-4 w-4" /></button>
          </div>
        </form>
        {error && <p role="alert" className="mt-3 rounded-xl bg-red-50 px-4 py-3 text-sm text-red-700">{error}</p>}
      </section>

      {!state ? <section className="grid max-w-4xl gap-4 md:grid-cols-3">
        {[['01', 'Plan', 'Breaks the question into focused, searchable lines of inquiry.'], ['02', 'Investigate', 'Searches, reads, and preserves source-grounded evidence.'], ['03', 'Synthesize', 'Writes a structured report with traceable numbered citations.']].map(item => <div key={item[0]} className="panel"><span className="text-xs font-bold text-moss">{item[0]}</span><h2 className="mt-5 font-semibold text-ink">{item[1]}</h2><p className="mt-2 text-sm leading-6 text-slate-500">{item[2]}</p></div>)}
      </section> : <div className="grid items-start gap-6 lg:grid-cols-[240px_minmax(0,1fr)_360px]">
        <aside className="sticky top-6 rounded-2xl border border-line bg-paper p-3">
          <div className="mb-3 flex items-center justify-between px-3 pt-2"><span className="eyebrow">Progress</span><span className="text-xs text-slate-400">Round {state.current_round || '—'}/2</span></div>
          <StatusRail status={state.status} round={state.current_round} />
          {state.demo_mode && <div className="mt-4 rounded-xl border border-amber-200 bg-amber-50 p-3 text-xs leading-5 text-amber-800"><strong>Demo mode</strong><br />Using deterministic synthesis and an embedded source set. Add API keys for live research.</div>}
        </aside>

        <section className="min-w-0 rounded-2xl border border-line bg-white p-6 shadow-soft lg:p-9">
          <div className="mb-8 border-b border-line pb-6"><div className="eyebrow">Research question</div><h1 className="mt-3 text-2xl font-semibold leading-9 tracking-tight text-ink">{state.original_question}</h1></div>
          {state.status === 'failed' ? <div className="rounded-xl bg-red-50 p-4 text-red-700"><strong>Research stopped</strong><p className="mt-1 text-sm">{state.error}</p></div>
            : state.final_report ? <article className="prose prose-slate max-w-none prose-headings:tracking-tight prose-headings:text-ink prose-a:text-moss prose-a:no-underline prose-strong:text-ink"><ReactMarkdown>{state.final_report}</ReactMarkdown></article>
            : <div className="grid min-h-[420px] place-items-center"><div className="text-center"><div className="mx-auto mb-4 h-10 w-10 animate-pulse rounded-2xl bg-emerald-100" /><p className="font-medium text-ink">{state.events.at(-1)?.message || 'Preparing research…'}</p><p className="mt-2 text-sm text-slate-400">Artifacts appear as each stage completes.</p></div></div>}
        </section>

        <aside><Artifacts state={state} /></aside>
      </div>}
    </div>
  </main>
}

