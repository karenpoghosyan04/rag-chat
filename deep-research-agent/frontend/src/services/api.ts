import type { ResearchState } from '../types/research'

export async function createResearch(question: string): Promise<string> {
  const response = await fetch('/api/research', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question }),
  })
  if (!response.ok) throw new Error((await response.json()).detail || 'Could not start research')
  return (await response.json()).task_id
}

export async function getResearch(taskId: string): Promise<ResearchState> {
  const response = await fetch(`/api/research/${taskId}`)
  if (!response.ok) throw new Error('Could not load research progress')
  return response.json()
}

