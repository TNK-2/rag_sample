// チャンク本文のうち、質問と共通する「トークン」に印を付ける。
// バックエンドの TF-IDF (rag/embedder.py の tokenize) と同じく、
// 日本語は文字 2-gram、英数字は単語単位で比較する。
// → 「なぜこのチャンクがヒットしたのか」が目で見て分かる。

export type Segment = { text: string; match: boolean }

const norm = (c: string) => c.normalize('NFKC').toLowerCase()
const isAsciiAlnum = (c: string) => /^[a-z0-9]$/.test(c)
const isWordChar = (c: string) => /^[\p{L}\p{N}]$/u.test(c) && !isAsciiAlnum(c)

function queryTokens(question: string): Set<string> {
  const tokens = new Set<string>()
  const q = question.normalize('NFKC').toLowerCase()
  for (const word of q.match(/[a-z0-9]+/g) ?? []) tokens.add(word)
  for (const span of q.match(/[\p{L}\p{N}]+/gu) ?? []) {
    const chars = [...span].filter((c) => !isAsciiAlnum(c))
    for (let i = 0; i + 1 < chars.length; i++) tokens.add(chars[i] + chars[i + 1])
  }
  return tokens
}

export function highlight(text: string, question: string): Segment[] {
  const tokens = queryTokens(question)
  const chars = [...text]
  const n = chars.map(norm)
  const marked = new Array<boolean>(chars.length).fill(false)

  for (let i = 0; i < chars.length; i++) {
    // 日本語など: 隣り合う 2 文字が質問の 2-gram に含まれていれば両方に印
    if (i + 1 < chars.length && isWordChar(n[i]) && isWordChar(n[i + 1]) && tokens.has(n[i] + n[i + 1])) {
      marked[i] = marked[i + 1] = true
    }
    // 英数字: 単語全体が一致したら印
    if (isAsciiAlnum(n[i]) && (i === 0 || !isAsciiAlnum(n[i - 1]))) {
      let j = i
      while (j < chars.length && isAsciiAlnum(n[j])) j++
      if (tokens.has(n.slice(i, j).join(''))) marked.fill(true, i, j)
    }
  }

  const segments: Segment[] = []
  chars.forEach((c, i) => {
    const last = segments[segments.length - 1]
    if (last && last.match === marked[i]) last.text += c
    else segments.push({ text: c, match: marked[i] })
  })
  return segments
}
