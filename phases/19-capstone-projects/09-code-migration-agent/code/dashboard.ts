/**
 * Code Migration Agent: dashboard skeleton (TypeScript).
 *
 * Mirrors the dashboard layer from the docs/en.md stack: an agent (Python) does
 * the migration in a sandbox, and a small dashboard renders progress for the
 * operator. This file is the dashboard. It serves a single HTML page plus two
 * JSON endpoints over the Node stdlib http module, simulates per-file diff
 * progress for a few in-flight repos, and exposes the schema an OpenRewrite +
 * libcst pipeline could fill in from real runs.
 *
 * Source: phases/19-capstone-projects/09-code-migration-agent/docs/en.md
 * Stack reference: MigrationBench harness, Moderne OpenRewrite, libcst.
 *
 * Runs on Node 20+ stdlib. No npm deps. No real API calls.
 */

import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { randomUUID } from "node:crypto";

type FileStatus = "queued" | "rewriting" | "building" | "passed" | "failed";

type FileDiff = {
  path: string;
  status: FileStatus;
  recipe: "openrewrite" | "libcst" | "agent";
  linesAdded: number;
  linesRemoved: number;
  testsTouched: number;
  lastError?: string;
};

type Migration = {
  id: string;
  repo: string;
  sourceRuntime: string;
  targetRuntime: string;
  startedAt: number;
  budgetUsd: number;
  spentUsd: number;
  turns: number;
  maxTurns: number;
  files: FileDiff[];
  state: "running" | "passed" | "failed" | "queued";
};

const MAX_TURNS = 20;
const BUDGET_USD = 8;

function seedMigration(
  repo: string,
  sourceRuntime: string,
  targetRuntime: string,
  files: FileDiff[],
): Migration {
  return {
    id: randomUUID(),
    repo,
    sourceRuntime,
    targetRuntime,
    startedAt: Date.now(),
    budgetUsd: BUDGET_USD,
    spentUsd: 0,
    turns: 0,
    maxTurns: MAX_TURNS,
    files,
    state: "running",
  };
}

function fileDiff(
  path: string,
  recipe: FileDiff["recipe"],
  status: FileStatus,
): FileDiff {
  return {
    path,
    status,
    recipe,
    linesAdded: 0,
    linesRemoved: 0,
    testsTouched: 0,
  };
}

const migrations: Migration[] = [
  seedMigration("acme/payments-svc", "java-8", "java-17", [
    fileDiff("pom.xml", "openrewrite", "queued"),
    fileDiff("src/main/java/Payments.java", "openrewrite", "queued"),
    fileDiff("src/main/java/Refunds.java", "openrewrite", "queued"),
    fileDiff("src/test/java/PaymentsTest.java", "agent", "queued"),
  ]),
  seedMigration("acme/billing-py", "python-2.7", "python-3.12", [
    fileDiff("setup.py", "libcst", "queued"),
    fileDiff("billing/core.py", "libcst", "queued"),
    fileDiff("billing/dunning.py", "agent", "queued"),
    fileDiff("tests/test_core.py", "libcst", "queued"),
  ]),
  seedMigration("acme/checkout-svc", "java-8", "java-17", [
    fileDiff("build.gradle", "openrewrite", "queued"),
    fileDiff("src/main/java/Checkout.java", "openrewrite", "queued"),
    fileDiff("src/main/java/Discount.java", "agent", "queued"),
  ]),
];

const STATE_ORDER: FileStatus[] = [
  "queued",
  "rewriting",
  "building",
  "passed",
];

function advanceFile(file: FileDiff): void {
  if (file.status === "passed" || file.status === "failed") return;
  const idx = STATE_ORDER.indexOf(file.status);
  const next = STATE_ORDER[idx + 1];
  if (!next) return;
  file.status = next;
  if (next === "rewriting") {
    file.linesAdded = 4 + Math.floor(Math.random() * 24);
    file.linesRemoved = 1 + Math.floor(Math.random() * 14);
  }
  if (next === "building" && Math.random() < 0.15) {
    file.status = "failed";
    file.lastError = "compile error: cannot find symbol javax.annotation.Nullable";
  }
  if (next === "passed" && file.path.includes("test")) {
    file.testsTouched = 2 + Math.floor(Math.random() * 6);
  }
}

function migrationDone(m: Migration): boolean {
  return m.files.every((f) => f.status === "passed" || f.status === "failed");
}

function tick(): void {
  for (const m of migrations) {
    if (m.state !== "running") continue;
    const inFlight = m.files.find(
      (f) => f.status !== "passed" && f.status !== "failed",
    );
    if (!inFlight) {
      m.state = m.files.some((f) => f.status === "failed") ? "failed" : "passed";
      continue;
    }
    advanceFile(inFlight);
    m.turns += 1;
    m.spentUsd = Number((m.spentUsd + 0.06 + Math.random() * 0.18).toFixed(3));
    if (m.spentUsd >= m.budgetUsd || m.turns >= m.maxTurns) {
      m.state = "failed";
    } else if (migrationDone(m)) {
      m.state = m.files.some((f) => f.status === "failed") ? "failed" : "passed";
    }
  }
}

function rolledUpStats(): {
  total: number;
  running: number;
  passed: number;
  failed: number;
  spentUsd: number;
} {
  let running = 0;
  let passed = 0;
  let failed = 0;
  let spent = 0;
  for (const m of migrations) {
    if (m.state === "running") running++;
    if (m.state === "passed") passed++;
    if (m.state === "failed") failed++;
    spent += m.spentUsd;
  }
  return {
    total: migrations.length,
    running,
    passed,
    failed,
    spentUsd: Number(spent.toFixed(3)),
  };
}

function renderDashboardHtml(): string {
  const stats = rolledUpStats();
  const rows = migrations
    .map((m) => {
      const passedFiles = m.files.filter((f) => f.status === "passed").length;
      const pct = Math.round((passedFiles / m.files.length) * 100);
      return [
        "<tr>",
        `<td><a href="/migrations/${m.id}">${m.repo}</a></td>`,
        `<td>${m.sourceRuntime} to ${m.targetRuntime}</td>`,
        `<td>${m.state}</td>`,
        `<td>${pct}%</td>`,
        `<td>${m.turns}/${m.maxTurns}</td>`,
        `<td>$${m.spentUsd.toFixed(2)}/$${m.budgetUsd}</td>`,
        "</tr>",
      ].join("");
    })
    .join("\n");
  return [
    "<!doctype html>",
    "<html><head><title>Code migration dashboard</title>",
    "<style>",
    "body{font-family:system-ui,sans-serif;margin:2rem;max-width:960px;}",
    "table{border-collapse:collapse;width:100%;}",
    "th,td{padding:.4rem .8rem;border-bottom:1px solid #ddd;text-align:left;}",
    "th{background:#f3f3f3;}",
    ".stats{display:flex;gap:1.5rem;margin-bottom:1rem;}",
    ".stat{background:#fafafa;border:1px solid #ddd;padding:.6rem 1rem;border-radius:6px;}",
    "</style></head><body>",
    "<h1>Code migration dashboard</h1>",
    "<div class='stats'>",
    `<div class='stat'><b>${stats.total}</b> migrations</div>`,
    `<div class='stat'>${stats.running} running</div>`,
    `<div class='stat'>${stats.passed} passed</div>`,
    `<div class='stat'>${stats.failed} failed</div>`,
    `<div class='stat'>$${stats.spentUsd.toFixed(2)} spent</div>`,
    "</div>",
    "<table><thead><tr>",
    "<th>repo</th><th>migration</th><th>state</th><th>progress</th><th>turns</th><th>cost</th>",
    "</tr></thead><tbody>",
    rows,
    "</tbody></table>",
    "<p><small>Auto-refreshes every 2s. Endpoints: /migrations, /migrations/:id.</small></p>",
    "<script>setTimeout(()=>location.reload(),2000)</script>",
    "</body></html>",
  ].join("\n");
}

function writeJson(res: ServerResponse, status: number, body: unknown): void {
  const payload = JSON.stringify(body, null, 2);
  res.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "content-length": Buffer.byteLength(payload),
  });
  res.end(payload);
}

function handle(req: IncomingMessage, res: ServerResponse): void {
  const url = new URL(req.url ?? "/", "http://localhost");
  const path = url.pathname;
  if (path === "/" || path === "/dashboard") {
    const html = renderDashboardHtml();
    res.writeHead(200, {
      "content-type": "text/html; charset=utf-8",
      "content-length": Buffer.byteLength(html),
    });
    res.end(html);
    return;
  }
  if (path === "/migrations") {
    writeJson(res, 200, {
      stats: rolledUpStats(),
      migrations: migrations.map((m) => ({
        id: m.id,
        repo: m.repo,
        state: m.state,
        sourceRuntime: m.sourceRuntime,
        targetRuntime: m.targetRuntime,
        turns: m.turns,
        spentUsd: m.spentUsd,
      })),
    });
    return;
  }
  const detail = path.match(/^\/migrations\/([0-9a-f-]+)$/);
  if (detail) {
    const m = migrations.find((x) => x.id === detail[1]);
    if (!m) {
      writeJson(res, 404, { error: "not_found", id: detail[1] });
      return;
    }
    writeJson(res, 200, m);
    return;
  }
  writeJson(res, 404, { error: "not_found", path });
}

function runDemoTicks(rounds: number): void {
  for (let i = 0; i < rounds; i++) tick();
}

function summarise(): void {
  const stats = rolledUpStats();
  console.log("[dashboard] migrations seeded:", migrations.length);
  for (const m of migrations) {
    const passed = m.files.filter((f) => f.status === "passed").length;
    console.log(
      `[dashboard] ${m.repo} ${m.sourceRuntime}->${m.targetRuntime} ` +
        `state=${m.state} files=${passed}/${m.files.length} ` +
        `turns=${m.turns}/${m.maxTurns} cost=$${m.spentUsd.toFixed(2)}`,
    );
  }
  console.log("[dashboard] roll-up:", stats);
}

function main(): void {
  console.log("[dashboard] simulating 40 ticks of agent progress...");
  runDemoTicks(40);
  summarise();
  if (process.env["SERVE"] === "1") {
    const port = Number(process.env["PORT"] ?? 8009);
    const server = createServer(handle);
    server.listen(port, () => {
      console.log(`[dashboard] serving on http://localhost:${port}`);
    });
    setInterval(tick, 750).unref();
  } else {
    console.log(
      "[dashboard] set SERVE=1 to start the HTTP dashboard on PORT (default 8009)",
    );
  }
}

main();
