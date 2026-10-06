import { Check, Circle, Loader2 } from 'lucide-react'
import type { Status } from '../types/research'

const stages: { key: Status; label: string }[] = [
  { key: 'planning', label: 'Planning research' },
  { key: 'searching', label: 'Searching sources' },
  { key: 'reading', label: 'Reading sources' },
  { key: 'extracting', label: 'Extracting evidence' },
  { key: 'analyzing_gaps', label: 'Identifying gaps' },
  { key: 'additional_research', label: 'Additional research' },
  { key: 'writing', label: 'Writing report' },
]

const order: Status[] = ['queued', ...stages.map(s => s.key), 'completed']

export function StatusRail({ status, round }: { status: Status; round: number }) {
  const current = order.indexOf(status)
  return <div className="space-y-1.5">
    {stages.map((stage) => {
      const index = order.indexOf(stage.key)
      const complete = status === 'completed' || current > index || (stage.key === 'additional_research' && round > 1)
      const active = status === stage.key
      return <div key={stage.key} className={`flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm ${active ? 'bg-white shadow-sm text-ink' : 'text-slate-500'}`}>
        {complete ? <Check className="h-4 w-4 text-moss" /> : active ? <Loader2 className="h-4 w-4 animate-spin text-moss" /> : <Circle className="h-3.5 w-3.5 text-slate-300" />}
        <span className={active ? 'font-semibold' : ''}>{stage.label}</span>
      </div>
    })}
  </div>
}

