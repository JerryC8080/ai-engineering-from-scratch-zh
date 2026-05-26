// Video understanding pipeline: TypeScript UI skeleton.
//
// The Python side ships the actual multi-vector index and temporal grounding
// (see code/main.py). This file is the UI half mentioned in the lesson stack
// ("Python (pipeline), TypeScript (UI)"). It is a stdlib HTTP server that
// exposes /jobs and /job/:id over the four pipeline stages a real video
// system actually runs: chunk, embed, index, qa.
//
// Source refs:
//   docs/en.md (this lesson)
//   VideoDB CRUD-for-video API:    https://videodb.io
//   TransNetV2 scene segmentation: https://github.com/soCzech/TransNetV2
//
// Run a self-terminating demo:  npx tsx main.ts --demo
// Run the HTTP server:          npx tsx main.ts --serve --port 8123

import { createServer, IncomingMessage, ServerResponse } from "node:http";

type Stage = "chunk" | "embed" | "index" | "qa";

type StageState = {
  stage: Stage;
  status: "pending" | "running" | "done" | "error";
  started_at?: number;
  finished_at?: number;
  detail?: string;
};

type Job = {
  id: string;
  video_url: string;
  question: string;
  created_at: number;
  stages: StageState[];
};

const STAGES: Stage[] = ["chunk", "embed", "index", "qa"];

const STAGE_DURATIONS_MS: Record<Stage, number> = {
  chunk: 1200,
  embed: 2400,
  index: 800,
  qa: 1600,
};

// In-memory fixture store. A real UI would talk to the Python pipeline over
// a queue or gRPC; here we mock the timeline so the dashboard renders.
const JOBS = new Map<string, Job>();

function newJob(id: string, video_url: string, question: string): Job {
  const created_at = Date.now();
  const stages: StageState[] = STAGES.map((stage) => ({ stage, status: "pending" }));
  const job: Job = { id, video_url, question, created_at, stages };
  JOBS.set(id, job);
  return job;
}

function advanceJob(job: Job, nowOverride?: number): void {
  const now = nowOverride ?? Date.now();
  let elapsed = now - job.created_at;
  for (const slot of job.stages) {
    const dur = STAGE_DURATIONS_MS[slot.stage];
    if (elapsed <= 0) {
      slot.status = "pending";
      continue;
    }
    if (elapsed < dur) {
      slot.status = "running";
      slot.started_at = slot.started_at ?? now - elapsed;
      slot.detail = `${Math.round((elapsed / dur) * 100)}% through ${slot.stage}`;
      break;
    }
    slot.status = "done";
    slot.started_at = slot.started_at ?? job.created_at;
    slot.finished_at = slot.started_at + dur;
    slot.detail = `${slot.stage} complete in ${dur}ms`;
    elapsed -= dur;
  }
}

function seedFixture(): void {
  const base = Date.now() - 8000;
  const j1 = newJob(
    "job-001",
    "vid_001",
    "how many cars pass through the intersection",
  );
  j1.created_at = base;
  advanceJob(j1);

  const j2 = newJob("job-002", "vid_001", "plating of the dish");
  j2.created_at = Date.now() - 3500;
  advanceJob(j2);

  const j3 = newJob("job-003", "vid_002", "ocean at sunset");
  // freshly queued, all pending
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

function sendHtml(res: ServerResponse, code: number, html: string): void {
  res.writeHead(code, {
    "content-type": "text/html; charset=utf-8",
    "content-length": Buffer.byteLength(html).toString(),
  });
  res.end(html);
}

function listJobs(): unknown {
  const items = [...JOBS.values()].map((j) => ({
    id: j.id,
    video_url: j.video_url,
    question: j.question,
    created_at: j.created_at,
    overall: overallStatus(j),
  }));
  items.sort((a, b) => b.created_at - a.created_at);
  return { jobs: items };
}

function overallStatus(j: Job): "pending" | "running" | "done" | "error" {
  if (j.stages.some((s) => s.status === "error")) return "error";
  if (j.stages.every((s) => s.status === "done")) return "done";
  if (j.stages.some((s) => s.status === "running")) return "running";
  return "pending";
}

function jobDetail(id: string): unknown | null {
  const job = JOBS.get(id);
  if (!job) return null;
  advanceJob(job);
  return {
    id: job.id,
    video_url: job.video_url,
    question: job.question,
    overall: overallStatus(job),
    timeline: job.stages.map((s) => ({
      stage: s.stage,
      status: s.status,
      started_at: s.started_at ?? null,
      finished_at: s.finished_at ?? null,
      detail: s.detail ?? null,
    })),
  };
}

function renderIndexHtml(): string {
  const rows = [...JOBS.values()]
    .sort((a, b) => b.created_at - a.created_at)
    .map(
      (j) =>
        `<tr><td>${j.id}</td><td>${j.video_url}</td><td>${j.question}</td><td>${overallStatus(j)}</td></tr>`,
    )
    .join("");
  return `<!doctype html><meta charset="utf-8"><title>video jobs</title>
<style>body{font-family:system-ui;margin:2rem}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:.4rem .6rem;text-align:left}</style>
<h1>video understanding jobs</h1>
<table><thead><tr><th>id</th><th>video</th><th>question</th><th>status</th></tr></thead>
<tbody>${rows}</tbody></table>
<p>JSON: <a href="/jobs">/jobs</a>, single job: <code>/job/&lt;id&gt;</code></p>`;
}

type Route = { method: string; pattern: RegExp };

const ROUTE_INDEX: Route = { method: "GET", pattern: /^\/$/ };
const ROUTE_JOBS: Route = { method: "GET", pattern: /^\/jobs\/?$/ };
const ROUTE_JOB: Route = { method: "GET", pattern: /^\/job\/([A-Za-z0-9_-]+)\/?$/ };

function handle(req: IncomingMessage, res: ServerResponse): void {
  const url = req.url ?? "/";
  if (req.method === ROUTE_INDEX.method && ROUTE_INDEX.pattern.test(url)) {
    sendHtml(res, 200, renderIndexHtml());
    return;
  }
  if (req.method === ROUTE_JOBS.method && ROUTE_JOBS.pattern.test(url)) {
    sendJson(res, 200, listJobs());
    return;
  }
  const m = url.match(ROUTE_JOB.pattern);
  if (req.method === ROUTE_JOB.method && m) {
    const body = jobDetail(m[1]);
    if (!body) {
      sendJson(res, 404, { error: "job not found", id: m[1] });
      return;
    }
    sendJson(res, 200, body);
    return;
  }
  sendJson(res, 404, { error: "no route", method: req.method, url });
}

function serve(port: number): void {
  seedFixture();
  const server = createServer(handle);
  server.listen(port, () => {
    process.stdout.write(`listening on http://localhost:${port}\n`);
  });
}

// --- self-terminating demo path ------------------------------------------

function demo(): void {
  seedFixture();
  process.stdout.write("=".repeat(72) + "\n");
  process.stdout.write("PHASE 19 LESSON 12 - video pipeline UI (TypeScript skeleton)\n");
  process.stdout.write("=".repeat(72) + "\n");

  process.stdout.write("\nGET /jobs\n");
  process.stdout.write(JSON.stringify(listJobs(), null, 2) + "\n");

  for (const id of ["job-001", "job-002", "job-003", "job-404"]) {
    process.stdout.write(`\nGET /job/${id}\n`);
    const body = jobDetail(id);
    if (!body) {
      process.stdout.write(JSON.stringify({ error: "not found", id }) + "\n");
      continue;
    }
    process.stdout.write(JSON.stringify(body, null, 2) + "\n");
  }

  // also verify the HTML rendering path resolves
  const html = renderIndexHtml();
  process.stdout.write(`\nrendered index html bytes: ${Buffer.byteLength(html)}\n`);
}

function main(): void {
  const argv = process.argv.slice(2);
  if (argv.includes("--serve")) {
    const portFlag = argv.indexOf("--port");
    const port = portFlag >= 0 ? Number(argv[portFlag + 1]) : 8123;
    serve(port);
    return;
  }
  demo();
}

main();
