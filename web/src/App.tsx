import { useEffect, useRef, useState } from 'react'
import * as api from './api'
import type { AnswerResponse, RetrieveResponse, Settings, Status } from './api'
import { highlight } from './highlight'

const EXAMPLES = [
  'リモートワークは週に何日まで？',
  'ソラマメ休暇の旅行補助はいくら？',
  '取引先との会食の上限金額は？',
  '本を買うのに会社はいくらまで出してくれる？',
  '社員食堂のメニューは？',
]

type Run = {
  question: string
  settings: Settings
  retrieval?: RetrieveResponse
  answer?: AnswerResponse
  error?: string
  loading: 'retrieve' | 'answer' | null
}

export default function App() {
  const [status, setStatus] = useState<Status | null>(null)
  const [question, setQuestion] = useState(EXAMPLES[0])
  const [settings, setSettings] = useState<Settings>({ top_k: 3, chunk_size: 300, overlap: 50, sources: null })
  const [generate, setGenerate] = useState(true)
  const [run, setRun] = useState<Run | null>(null)
  const [activeCitation, setActiveCitation] = useState<number | null>(null)
  const runId = useRef(0)

  useEffect(() => {
    api.getStatus().then(setStatus).catch(() => setStatus(null))
  }, [])

  const update = (id: number, patch: Partial<Run>) => {
    // 古いリクエストの結果が後から返ってきても無視する
    if (id === runId.current) setRun((r) => (r ? { ...r, ...patch } : r))
  }

  async function submit(q = question) {
    const text = q.trim()
    if (!text) return
    const id = ++runId.current
    const snapshot = { ...settings }
    setActiveCitation(null)
    setRun({ question: text, settings: snapshot, loading: 'retrieve' })
    try {
      // STEP 1: まず検索だけ (速い)。結果をすぐ表示する
      const retrieval = await api.retrieve(text, snapshot)
      update(id, { retrieval, loading: generate ? 'answer' : null })
      if (!generate) return
      // STEP 2: Claude に回答を生成させる (数秒かかる)
      update(id, { answer: await api.answer(text, snapshot), loading: null })
    } catch (e) {
      update(id, { error: e instanceof Error ? e.message : String(e), loading: null })
    }
  }

  const maxOverlap = (chunkSize: number) => Math.min(200, chunkSize - 50)
  const setNum = (key: 'top_k' | 'chunk_size' | 'overlap') => (e: React.ChangeEvent<HTMLInputElement>) =>
    setSettings((s) => {
      const next = { ...s, [key]: Number(e.target.value) }
      // overlap は chunk_size より小さくないといけないので、chunk_size を縮めたら追従させる
      return { ...next, overlap: Math.min(next.overlap, maxOverlap(next.chunk_size)) }
    })

  const allSources = status?.documents.map((d) => d.source) ?? []
  const isSelected = (source: string) => settings.sources === null || settings.sources.includes(source)
  const toggleSource = (source: string) =>
    setSettings((s) => {
      const current = s.sources ?? allSources
      const next = current.includes(source) ? current.filter((x) => x !== source) : [...current, source]
      // 全部選ばれていれば絞り込みなし (null) として送る
      return { ...s, sources: next.length === allSources.length ? null : next }
    })

  return (
    <div className="layout">
      <aside className="sidebar">
        <h1>
          RAG Playground
          <small>作りながら学ぶ RAG</small>
        </h1>

        <section>
          <h2>検索の設定</h2>
          <Slider label="top_k" hint="LLM に渡すチャンク数" min={1} max={8} step={1}
            value={settings.top_k} onChange={setNum('top_k')} />
          <Slider label="chunk_size" hint="1 チャンクの目安文字数" min={100} max={1000} step={50}
            value={settings.chunk_size} onChange={setNum('chunk_size')} />
          <Slider label="overlap" hint="隣のチャンクと重ねる文字数" min={0} max={maxOverlap(settings.chunk_size)} step={10}
            value={settings.overlap} onChange={setNum('overlap')} />
          <label className="toggle">
            <input type="checkbox" checked={generate} onChange={(e) => setGenerate(e.target.checked)} />
            <span>Claude で回答を生成する</span>
          </label>
          <p className="note">オフにすると検索だけ行います (API キー不要)。</p>
        </section>

        <section>
          <h2>検索対象の文書</h2>
          {status ? (
            <>
              <ul className="docs">
                {status.documents.map((d) => (
                  <li key={d.source}>
                    <label>
                      <input type="checkbox" checked={isSelected(d.source)} onChange={() => toggleSource(d.source)} />
                      <span>
                        {d.title}
                        <code>{d.source}</code>
                      </span>
                    </label>
                  </li>
                ))}
              </ul>
              <p className="note">チェックを外すと、ベクトル DB のメタデータフィルタでその文書を検索対象から外します。</p>
            </>
          ) : (
            <p className="note">API サーバーに接続できません。<code>uvicorn api:app --port 8000</code> を起動してください。</p>
          )}
        </section>

        {status && (
          <section>
            <h2>接続先</h2>
            <dl className="info">
              <dt>ベクトル DB</dt>
              <dd>{status.store.kind === 'qdrant' ? 'Qdrant' : status.store.kind}<code>{status.store.location}</code></dd>
              <dt>LLM</dt>
              <dd><code>{status.model}</code></dd>
            </dl>
          </section>
        )}
      </aside>

      <main>
        <form className="ask" onSubmit={(e) => { e.preventDefault(); submit() }}>
          <input value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="社内規程について質問する…" />
          <button type="submit" disabled={run?.loading != null}>質問する</button>
        </form>
        <div className="examples">
          {EXAMPLES.map((ex) => (
            <button key={ex} type="button" onClick={() => { setQuestion(ex); submit(ex) }} disabled={run?.loading != null}>
              {ex}
            </button>
          ))}
        </div>

        {!run && <Intro />}
        {run && <Result run={run} generate={generate} activeCitation={activeCitation} onCite={setActiveCitation} />}
      </main>
    </div>
  )
}

function Slider(props: {
  label: string; hint: string; min: number; max: number; step: number; value: number
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void
}) {
  return (
    <label className="slider">
      <div><code>{props.label}</code><b>{props.value}</b></div>
      <input type="range" min={props.min} max={props.max} step={props.step} value={props.value} onChange={props.onChange} />
      <span className="note">{props.hint}</span>
    </label>
  )
}

function Intro() {
  return (
    <div className="intro">
      <p>質問すると、RAG の各ステップで何が起きているかを順に表示します。</p>
      <ol>
        <li><b>検索</b>: 質問に似たチャンクを TF-IDF + コサイン類似度で探す</li>
        <li><b>プロンプト</b>: 見つけたチャンクを資料として質問と一緒に詰める</li>
        <li><b>回答</b>: Claude が資料だけを根拠に答える</li>
      </ol>
      <p className="note">
        「本を買うのに…」は資料の「書籍購入」と共通する文字がないため、TF-IDF では見つかりません。
        「社員食堂のメニューは？」は資料に答えがない質問です。
      </p>
    </div>
  )
}

function Result({ run, generate, activeCitation, onCite }: {
  run: Run; generate: boolean; activeCitation: number | null; onCite: (n: number | null) => void
}) {
  const { retrieval, answer, error, loading } = run
  return (
    <div className="steps">
      <Step n={1} title="検索 (Retrieval)"
        meta={retrieval && `コレクション ${retrieval.collection} (${retrieval.num_chunks} チャンク) から上位 ${run.settings.top_k} 件${run.settings.sources ? ` ・ ${run.settings.sources.length} 文書に絞り込み` : ''}`}>
        {loading === 'retrieve' && <Loading text="検索中…" />}
        {retrieval && retrieval.results.length === 0 && (
          <p className="empty">
            {run.settings.sources?.length === 0
              ? '検索対象の文書が 1 つも選ばれていません。'
              : '関連するチャンクが見つかりませんでした (スコアが足切り値以下)。'}
          </p>
        )}
        {retrieval && retrieval.results.length > 0 && (
          <>
            <p className="note">
              <mark>ハイライト</mark> は質問と共通する文字 2-gram。TF-IDF はこの重なりでスコアを計算しています。
            </p>
            <div className="chunks">
              {retrieval.results.map((r, i) => (
                <article key={r.id} id={`chunk-${i + 1}`} className={activeCitation === i + 1 ? 'chunk active' : 'chunk'}>
                  <header>
                    <span className="rank">[{i + 1}]</span>
                    <span className="title">{r.title}</span>
                    <code>{r.id}</code>
                  </header>
                  <div className="score">
                    <div className="bar"><div style={{ width: `${Math.max(2, r.score * 100)}%` }} /></div>
                    <span>{r.score.toFixed(3)}</span>
                  </div>
                  <p className="chunk-text">
                    {highlight(r.text, run.question).map((s, j) => (s.match ? <mark key={j}>{s.text}</mark> : s.text))}
                  </p>
                </article>
              ))}
            </div>
          </>
        )}
      </Step>

      {retrieval && (
        <Step n={2} title="プロンプト (Augmentation)" meta="この内容が Claude に送られます">
          <details>
            <summary>プロンプトを表示</summary>
            <pre>{retrieval.prompt}</pre>
          </details>
        </Step>
      )}

      {(retrieval || error) && (generate || error) && (
        <Step n={3} title="回答 (Generation)" meta={answer && answer.model}>
          {loading === 'answer' && <Loading text="Claude が回答を生成中…" />}
          {error && <p className="error">{error}</p>}
          {answer && <Answer text={answer.answer} onCite={onCite} />}
        </Step>
      )}
    </div>
  )
}

function Answer({ text, onCite }: { text: string; onCite: (n: number | null) => void }) {
  // 回答中の [1] などの出典番号をボタンにして、対応するチャンクを強調表示する
  const parts = text.split(/(\[\d+\])/g)
  return (
    <div className="answer">
      {parts.map((p, i) => {
        const m = p.match(/^\[(\d+)\]$/)
        if (!m) return <span key={i}>{p}</span>
        const n = Number(m[1])
        return (
          <button key={i} type="button" className="cite"
            onMouseEnter={() => onCite(n)} onMouseLeave={() => onCite(null)}
            onClick={() => document.getElementById(`chunk-${n}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' })}>
            {n}
          </button>
        )
      })}
    </div>
  )
}

function Step({ n, title, meta, children }: { n: number; title: string; meta?: string | false; children: React.ReactNode }) {
  return (
    <section className="step">
      <h2><span className="step-n">{n}</span>{title}{meta && <small>{meta}</small>}</h2>
      {children}
    </section>
  )
}

function Loading({ text }: { text: string }) {
  return <p className="loading"><span className="spinner" />{text}</p>
}
