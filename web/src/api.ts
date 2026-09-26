export type Settings = {
  top_k: number
  chunk_size: number
  overlap: number
}

export type ChunkResult = {
  id: string
  source: string
  title: string
  text: string
  score: number
}

export type RetrieveResponse = {
  results: ChunkResult[]
  prompt: string
  num_chunks: number
}

export type AnswerResponse = {
  answer: string
  model: string
}

export type Status = {
  model: string
  documents: { source: string; title: string }[]
}

async function request<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, {
    method: body === undefined ? 'GET' : 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
  if (!res.ok) {
    const data = await res.json().catch(() => null)
    const detail = data?.detail
    throw new Error(typeof detail === 'string' ? detail : `サーバーエラー (${res.status})`)
  }
  return res.json()
}

export const getStatus = () => request<Status>('/api/status')

export const retrieve = (question: string, settings: Settings) =>
  request<RetrieveResponse>('/api/retrieve', { question, ...settings })

export const answer = (question: string, settings: Settings) =>
  request<AnswerResponse>('/api/answer', { question, ...settings })
