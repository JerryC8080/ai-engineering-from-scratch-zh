// Personal AI Tutor: web app skeleton (TypeScript, stdlib only).
//
// The Python side (code/main.py) ships the learner model and tutor policy.
// This file is the "TypeScript (web app)" half of the stack: a stdlib HTTP
// server that exposes /lesson/next (topo-walk over a curriculum DAG fixture)
// and /lesson/:id/submit (records mastery + schedules next review).
//
// Source refs:
//   docs/en.md (this lesson)
//   Bayesian Knowledge Tracing: https://en.wikipedia.org/wiki/Bayesian_knowledge_tracing
//   FSRS spaced-repetition:     https://github.com/open-spaced-repetition/fsrs4anki
//
// Self-terminating demo (default): npx tsx main.ts
// Live HTTP server:                npx tsx main.ts --serve --port 8090

import { createServer, IncomingMessage, ServerResponse } from "node:http";

// --- Curriculum DAG fixture (intro Python subset) -------------------------

type Lesson = { id: string; title: string; prereqs: string[] };

const CURRICULUM: Lesson[] = [
  { id: "py-01", title: "variables and types", prereqs: [] },
  { id: "py-02", title: "arithmetic operators", prereqs: ["py-01"] },
  { id: "py-03", title: "strings", prereqs: ["py-01"] },
  { id: "py-04", title: "if / else", prereqs: ["py-02"] },
  { id: "py-05", title: "for loops", prereqs: ["py-04"] },
  { id: "py-06", title: "lists", prereqs: ["py-03", "py-05"] },
  { id: "py-07", title: "dicts", prereqs: ["py-06"] },
  { id: "py-08", title: "functions", prereqs: ["py-04"] },
  { id: "py-09", title: "list comprehensions", prereqs: ["py-06", "py-08"] },
];

const LESSON_INDEX: Record<string, Lesson> = Object.fromEntries(
  CURRICULUM.map((l) => [l.id, l]),
);

// Kahn topological sort, used to pick the next eligible node whose
// prereqs are already mastered.

function topoOrder(items: Lesson[]): string[] {
  const indeg: Record<string, number> = {};
  const out: Record<string, string[]> = {};
  for (const l of items) {
    indeg[l.id] = indeg[l.id] ?? 0;
    out[l.id] = out[l.id] ?? [];
    for (const p of l.prereqs) {
      indeg[l.id] = (indeg[l.id] ?? 0) + 1;
      out[p] = out[p] ?? [];
      out[p].push(l.id);
    }
  }
  const ready: string[] = [];
  for (const id of Object.keys(indeg)) if (indeg[id] === 0) ready.push(id);
  ready.sort();
  const order: string[] = [];
  while (ready.length > 0) {
    const id = ready.shift() as string;
    order.push(id);
    for (const nxt of out[id] ?? []) {
      indeg[nxt] -= 1;
      if (indeg[nxt] === 0) {
        ready.push(nxt);
        ready.sort();
      }
    }
  }
  return order;
}

const TOPO = topoOrder(CURRICULUM);

// --- Learner model --------------------------------------------------------
//
// Per-lesson mastery score (0..1). Spaced-repetition uses a simple half-life
// doubling schedule (FSRS-lite): each successful review doubles the next-due
// interval; a failure halves it.

type Mastery = {
  score: number;
  attempts: number;
  successes: number;
  next_due_at: number;
  interval_ms: number;
};

const MASTERY: Record<string, Mastery> = {};
const MASTERY_THRESHOLD = 0.7;
const BASE_INTERVAL_MS = 1000 * 60 * 60 * 24; // 1 day

function getMastery(id: string): Mastery {
  let m = MASTERY[id];
  if (!m) {
    m = { score: 0, attempts: 0, successes: 0, next_due_at: 0, interval_ms: BASE_INTERVAL_MS };
    MASTERY[id] = m;
  }
  return m;
}

function recordOutcome(id: string, correct: boolean, nowOverride?: number): Mastery {
  const m = getMastery(id);
  const now = nowOverride ?? Date.now();
  m.attempts += 1;
  if (correct) {
    m.successes += 1;
    m.interval_ms = Math.min(m.interval_ms * 2, BASE_INTERVAL_MS * 30);
  } else {
    m.interval_ms = Math.max(Math.floor(m.interval_ms / 2), 60_000);
  }
  // simple BKT-like update: blend prior + observed success rate
  const observed = m.successes / m.attempts;
  m.score = 0.3 * m.score + 0.7 * observed;
  m.next_due_at = now + m.interval_ms;
  return m;
}

function nextLesson(now: number): { lesson: Lesson; reason: string } | null {
  // priority 1: a new un-mastered lesson whose prereqs are all mastered.
  // Picking new work over reviews keeps the learner advancing; overdue
  // reviews surface once the frontier has no eligible new nodes.
  for (const id of TOPO) {
    const m = MASTERY[id];
    const mastered = (m?.score ?? 0) >= MASTERY_THRESHOLD;
    if (mastered) continue;
    const prereqsMet = LESSON_INDEX[id].prereqs.every(
      (p) => (MASTERY[p]?.score ?? 0) >= MASTERY_THRESHOLD,
    );
    if (prereqsMet) return { lesson: LESSON_INDEX[id], reason: "new_eligible" };
  }
  // priority 2: anything overdue for review and not yet at ceiling
  const overdue = TOPO
    .filter((id) => {
      const m = MASTERY[id];
      return m && m.attempts > 0 && m.next_due_at <= now && m.score < 0.95;
    })
    .map((id) => LESSON_INDEX[id]);
  if (overdue.length > 0) return { lesson: overdue[0], reason: "review_overdue" };
  return null;
}

// --- HTTP layer -----------------------------------------------------------

function sendJson(res: ServerResponse, code: number, body: unknown): void {
  const payload = JSON.stringify(body);
  res.writeHead(code, {
    "content-type": "application/json; charset=utf-8",
    "content-length": Buffer.byteLength(payload).toString(),
  });
  res.end(payload);
}

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on("data", (c: Buffer) => chunks.push(c));
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = req.url ?? "/";
  if (req.method === "GET" && url === "/lesson/next") {
    const pick = nextLesson(Date.now());
    if (!pick) {
      sendJson(res, 200, { done: true, message: "curriculum complete" });
      return;
    }
    sendJson(res, 200, {
      lesson: pick.lesson,
      reason: pick.reason,
      mastery: MASTERY[pick.lesson.id] ?? null,
    });
    return;
  }
  const m = url.match(/^\/lesson\/([A-Za-z0-9_-]+)\/submit\/?$/);
  if (req.method === "POST" && m) {
    const id = m[1];
    if (!LESSON_INDEX[id]) {
      sendJson(res, 404, { error: "unknown lesson", id });
      return;
    }
    const raw = await readBody(req);
    let parsed: { correct?: unknown } = {};
    try {
      parsed = raw ? (JSON.parse(raw) as { correct?: unknown }) : {};
    } catch (err) {
      sendJson(res, 400, { error: "invalid JSON", detail: String(err) });
      return;
    }
    const correct = parsed.correct === true;
    const updated = recordOutcome(id, correct);
    sendJson(res, 200, { id, correct, mastery: updated });
    return;
  }
  sendJson(res, 404, { error: "no route", method: req.method, url });
}

function serve(port: number): void {
  const server = createServer((req, res) => {
    handle(req, res).catch((err) => sendJson(res, 500, { error: String(err) }));
  });
  server.listen(port, () => {
    process.stdout.write(`tutor api on http://localhost:${port}\n`);
  });
}

// --- self-terminating demo -----------------------------------------------
//
// Walk the curriculum: ask for next lesson, simulate a learner answering,
// repeat until we hit completion or a fixed bound. Drives the topo sort and
// spaced-rep timestamps without opening a socket.

function demo(): void {
  process.stdout.write("=".repeat(72) + "\n");
  process.stdout.write("PHASE 19 LESSON 17 - personal tutor web app (TypeScript)\n");
  process.stdout.write("=".repeat(72) + "\n");

  process.stdout.write(`\ntopological order: ${TOPO.join(", ")}\n`);

  let now = Date.now();
  const learnerCorrectRate = 0.75;
  // deterministic pseudo-RNG so the demo is reproducible
  let seed = 1;
  const rng = (): number => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x7fffffff;
  };

  for (let step = 0; step < 14; step += 1) {
    const pick = nextLesson(now);
    if (!pick) {
      process.stdout.write(`\nstep ${step}: curriculum complete\n`);
      break;
    }
    const correct = rng() < learnerCorrectRate;
    const updated = recordOutcome(pick.lesson.id, correct, now);
    process.stdout.write(
      `\nstep ${step}: ${pick.lesson.id} (${pick.lesson.title}) ${pick.reason}, ` +
        `learner ${correct ? "correct" : "wrong"}, ` +
        `score=${updated.score.toFixed(2)}, next_due=+${Math.floor(updated.interval_ms / 1000)}s\n`,
    );
    // advance simulated clock past the just-set next_due so reviews surface
    now = updated.next_due_at + 1;
  }

  process.stdout.write("\nfinal mastery snapshot:\n");
  for (const id of TOPO) {
    const m = MASTERY[id];
    if (!m) continue;
    process.stdout.write(
      `  ${id}: score=${m.score.toFixed(2)} attempts=${m.attempts} successes=${m.successes}\n`,
    );
  }
}

function main(): void {
  if (process.argv.includes("--serve")) {
    const argv = process.argv.slice(2);
    const portFlag = argv.indexOf("--port");
    const port = portFlag >= 0 ? Number(argv[portFlag + 1]) : 8090;
    serve(port);
    return;
  }
  demo();
}

main();
