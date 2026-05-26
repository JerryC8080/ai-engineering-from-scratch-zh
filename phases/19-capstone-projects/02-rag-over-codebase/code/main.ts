// Capstone 19/02: code RAG query API skeleton (TypeScript).
//
// Sources:
//   This lesson's docs/en.md (hybrid retrieval + cited answer API)
//   node:http reference         https://nodejs.org/api/http.html
//   BM25 (Robertson + Zaragoza) https://en.wikipedia.org/wiki/Okapi_BM25
//   Reciprocal Rank Fusion       https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
//
// The API + UI side of the capstone: an in-memory hybrid index (dense cosine over
// hash-trick embeddings, plus BM25), RRF merge, /query and /healthz endpoints
// served by node:http. Returns JSON answers with file:line citations. No
// external deps, no network calls, deterministic for testing.
//
// Run: npx -y tsx@4.19.2 code/main.ts

import * as http from "node:http";

type Chunk = {
  repo: string;
  path: string;
  startLine: number;
  endLine: number;
  symbol: string;
  body: string;
  summary: string;
};

function anchor(c: Chunk): string {
  return `${c.repo}/${c.path}:${c.startLine}-${c.endLine}`;
}

const SAMPLE_CORPUS: Chunk[] = [
  {
    repo: "uploader",
    path: "services/retry.go",
    startLine: 122,
    endLine: 148,
    symbol: "AbortMultipartOnFail",
    body: "if ctx.Err() != nil { return abort() }; decrement bucket budget; retry with backoff",
    summary:
      "aborts an in-flight S3 multipart upload and decrements the per-bucket retry budget",
  },
  {
    repo: "uploader",
    path: "config/budgets.yaml",
    startLine: 34,
    endLine: 51,
    symbol: "bucket_budget",
    body: "per_bucket_budget: 64; backoff_ms: [100, 500, 2500]; abort_threshold: 3",
    summary:
      "declares the retry budget and exponential backoff schedule per S3 bucket",
  },
  {
    repo: "client",
    path: "libs/s3client/multipart.ts",
    startLine: 44,
    endLine: 61,
    symbol: "abortUpload",
    body: "await s3.abortMultipartUpload({Bucket, Key, UploadId}); metrics.inc('s3.abort')",
    summary: "client-side S3 multipart abort with metrics instrumentation",
  },
  {
    repo: "auth",
    path: "services/authz/check.py",
    startLine: 12,
    endLine: 38,
    symbol: "check_permission",
    body: "def check_permission(user, resource, action): return policy.evaluate(user, resource, action)",
    summary:
      "central authorization gateway evaluating an OPA policy for user-resource-action",
  },
  {
    repo: "auth",
    path: "libs/policy/opa.py",
    startLine: 88,
    endLine: 110,
    symbol: "evaluate",
    body: "def evaluate(user, resource, action): return self.engine.query('authz', input=...)",
    summary: "OPA policy engine query wrapper for authorization checks",
  },
  {
    repo: "catalog",
    path: "services/search/query.rs",
    startLine: 200,
    endLine: 240,
    symbol: "rank_fusion",
    body: "pub fn rank_fusion(dense: Vec<Hit>, sparse: Vec<Hit>) -> Vec<Hit>",
    summary: "reciprocal rank fusion of dense and sparse retrieval results",
  },
];

const TOKEN_RE = /[a-z0-9_]+/g;

function tokenize(text: string): string[] {
  return text.toLowerCase().match(TOKEN_RE) ?? [];
}

// Tiny deterministic 32-bit hash (FNV-1a) so embeddings are stable across runs.
function fnv1a(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

function fakeEmbed(text: string, dim = 64): number[] {
  // Hash-trick embedder: a deterministic stand-in for Voyage-code-3 / bge-code.
  // Real ingestion would write vectors to Qdrant/pgvector; here it stays in RAM.
  const vec = new Array<number>(dim).fill(0);
  for (const tok of tokenize(text)) {
    const h = fnv1a(tok);
    vec[h % dim] += 1.0;
    vec[(h >>> 8) % dim] += 0.5;
  }
  let norm = 0;
  for (const v of vec) norm += v * v;
  norm = Math.sqrt(norm) || 1.0;
  return vec.map((v) => v / norm);
}

function cosine(a: readonly number[], b: readonly number[]): number {
  let s = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) s += a[i] * b[i];
  return s;
}

class DenseIndex {
  private vectors: { chunk: Chunk; vec: number[] }[] = [];

  add(chunk: Chunk): void {
    const text = `${chunk.symbol}\n${chunk.summary}\n${chunk.body}`;
    this.vectors.push({ chunk, vec: fakeEmbed(text) });
  }

  search(query: string, k = 10): { chunk: Chunk; score: number }[] {
    const qv = fakeEmbed(query);
    const scored = this.vectors.map((v) => ({
      chunk: v.chunk,
      score: cosine(qv, v.vec),
    }));
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, k);
  }

  size(): number {
    return this.vectors.length;
  }
}

class BM25Index {
  k1 = 1.5;
  b = 0.75;
  private docs: Chunk[] = [];
  private docLens: number[] = [];
  private df = new Map<string, number>();
  private tf: Map<string, number>[] = [];
  private avgdl = 0;

  add(chunk: Chunk): void {
    // Field-weighted tokenization: symbol x4, summary x2, body x1.
    const repeat = (toks: string[], times: number): string[] => {
      const out: string[] = [];
      for (let i = 0; i < times; i++) out.push(...toks);
      return out;
    };
    const tokens = [
      ...repeat(tokenize(chunk.symbol), 4),
      ...repeat(tokenize(chunk.summary), 2),
      ...tokenize(chunk.body),
    ];
    const counts = new Map<string, number>();
    for (const t of tokens) counts.set(t, (counts.get(t) ?? 0) + 1);
    this.docs.push(chunk);
    this.docLens.push(tokens.length);
    this.tf.push(counts);
    for (const term of counts.keys()) {
      this.df.set(term, (this.df.get(term) ?? 0) + 1);
    }
    this.avgdl = this.docLens.reduce((s, n) => s + n, 0) / this.docLens.length;
  }

  search(query: string, k = 10): { chunk: Chunk; score: number }[] {
    const qTerms = tokenize(query);
    const n = this.docs.length;
    const scores = new Array<number>(n).fill(0);
    for (const term of qTerms) {
      const df = this.df.get(term);
      if (!df) continue;
      const idf = Math.log((n - df + 0.5) / (df + 0.5) + 1.0);
      for (let i = 0; i < n; i++) {
        const f = this.tf[i].get(term) ?? 0;
        if (!f) continue;
        const dl = this.docLens[i];
        const denom = f + this.k1 * (1 - this.b + (this.b * dl) / this.avgdl);
        scores[i] += (idf * f * (this.k1 + 1)) / denom;
      }
    }
    const ranked = this.docs
      .map((chunk, i) => ({ chunk, score: scores[i] }))
      .filter((r) => r.score > 0);
    ranked.sort((a, b) => b.score - a.score);
    return ranked.slice(0, k);
  }
}

function rrf(
  dense: { chunk: Chunk; score: number }[],
  sparse: { chunk: Chunk; score: number }[],
  kRrf = 60,
): { chunk: Chunk; score: number }[] {
  const scoreByAnchor = new Map<string, number>();
  const byAnchor = new Map<string, Chunk>();
  dense.forEach(({ chunk }, rank) => {
    const a = anchor(chunk);
    scoreByAnchor.set(a, (scoreByAnchor.get(a) ?? 0) + 1.0 / (kRrf + rank + 1));
    byAnchor.set(a, chunk);
  });
  sparse.forEach(({ chunk }, rank) => {
    const a = anchor(chunk);
    scoreByAnchor.set(a, (scoreByAnchor.get(a) ?? 0) + 1.0 / (kRrf + rank + 1));
    byAnchor.set(a, chunk);
  });
  const fused = [...scoreByAnchor.entries()].sort((a, b) => b[1] - a[1]);
  return fused.map(([a, score]) => ({ chunk: byAnchor.get(a)!, score }));
}

type QueryResponse = {
  query: string;
  denseTop: string[];
  sparseTop: string[];
  fusedTop: string[];
  citations: { anchor: string; score: number }[];
};

function runQuery(
  q: string,
  dense: DenseIndex,
  bm25: BM25Index,
  topK = 5,
): QueryResponse {
  const d = dense.search(q, 10);
  const s = bm25.search(q, 10);
  const fused = rrf(d, s);
  const top = fused.slice(0, topK);
  return {
    query: q,
    denseTop: d.slice(0, 3).map((r) => anchor(r.chunk)),
    sparseTop: s.slice(0, 3).map((r) => anchor(r.chunk)),
    fusedTop: fused.slice(0, 5).map((r) => anchor(r.chunk)),
    citations: top.map((r) => ({ anchor: anchor(r.chunk), score: r.score })),
  };
}

function buildIndices(): { dense: DenseIndex; bm25: BM25Index } {
  const dense = new DenseIndex();
  const bm25 = new BM25Index();
  for (const c of SAMPLE_CORPUS) {
    dense.add(c);
    bm25.add(c);
  }
  return { dense, bm25 };
}

function readBody(req: http.IncomingMessage, max = 64 * 1024): Promise<string> {
  return new Promise((resolve, reject) => {
    let size = 0;
    const chunks: Buffer[] = [];
    req.on("data", (chunk: Buffer) => {
      size += chunk.length;
      if (size > max) {
        reject(new Error("payload too large"));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}

function send(res: http.ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "content-length": Buffer.byteLength(text),
  });
  res.end(text);
}

function buildServer(dense: DenseIndex, bm25: BM25Index): http.Server {
  return http.createServer(async (req, res) => {
    const url = new URL(req.url ?? "/", "http://localhost");
    if (req.method === "GET" && url.pathname === "/healthz") {
      send(res, 200, { ok: true, corpus: SAMPLE_CORPUS.length });
      return;
    }
    if (req.method === "GET" && url.pathname === "/query") {
      const q = url.searchParams.get("q");
      if (!q) {
        send(res, 400, { error: "missing 'q' query parameter" });
        return;
      }
      send(res, 200, runQuery(q, dense, bm25));
      return;
    }
    if (req.method === "POST" && url.pathname === "/query") {
      try {
        const raw = await readBody(req);
        const parsed = raw ? (JSON.parse(raw) as { q?: unknown; topK?: unknown }) : {};
        const q = typeof parsed.q === "string" ? parsed.q : "";
        const topK = typeof parsed.topK === "number" ? parsed.topK : 5;
        if (!q) {
          send(res, 400, { error: "missing 'q' in request body" });
          return;
        }
        send(res, 200, runQuery(q, dense, bm25, topK));
      } catch (err) {
        send(res, 400, { error: (err as Error).message });
      }
      return;
    }
    send(res, 404, { error: "not found", path: url.pathname });
  });
}

async function probe(server: http.Server, port: number): Promise<void> {
  const queries = [
    "how is S3 multipart abort wired into retry budget",
    "where is authorization centralized",
    "how does rank fusion work",
  ];
  const get = (path: string): Promise<{ status: number; body: string }> =>
    new Promise((resolve, reject) => {
      const req = http.request(
        { host: "127.0.0.1", port, path, method: "GET" },
        (resp) => {
          const parts: Buffer[] = [];
          resp.on("data", (c: Buffer) => parts.push(c));
          resp.on("end", () =>
            resolve({
              status: resp.statusCode ?? 0,
              body: Buffer.concat(parts).toString("utf8"),
            }),
          );
        },
      );
      req.on("error", reject);
      req.end();
    });

  const health = await get("/healthz");
  console.log(`GET /healthz -> ${health.status} ${health.body}`);
  for (const q of queries) {
    const r = await get(`/query?q=${encodeURIComponent(q)}`);
    const parsed = JSON.parse(r.body) as QueryResponse;
    console.log(`GET /query?q=${JSON.stringify(q)} -> ${r.status}`);
    console.log(`  dense  : ${JSON.stringify(parsed.denseTop)}`);
    console.log(`  sparse : ${JSON.stringify(parsed.sparseTop)}`);
    console.log(`  fused  : ${JSON.stringify(parsed.fusedTop)}`);
    console.log(
      `  cites  : ${parsed.citations
        .map((c) => `${c.anchor}@${c.score.toFixed(4)}`)
        .join(", ")}`,
    );
  }
  await new Promise<void>((resolve) => server.close(() => resolve()));
}

async function main(): Promise<void> {
  const { dense, bm25 } = buildIndices();
  console.log(`indexed ${dense.size()} chunks across ${SAMPLE_CORPUS.length} entries`);
  const server = buildServer(dense, bm25);
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", () => resolve()));
  const addr = server.address();
  if (!addr || typeof addr === "string") throw new Error("server address unavailable");
  const port = addr.port;
  console.log(`code-rag api listening on http://127.0.0.1:${port}`);
  if (process.argv.includes("--serve")) {
    process.on("SIGINT", () => server.close(() => process.exit(0)));
    return;
  }
  await probe(server, port);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
