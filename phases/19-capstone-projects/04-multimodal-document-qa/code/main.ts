// Capstone 04 - Multimodal Document QA viewer UI skeleton, TypeScript stdlib only.
//
// Source: phases/19-capstone-projects/04-multimodal-document-qa/docs/en.md
//   "Viewer UI: Next.js 15 with canvas overlay for evidence regions"
//   "Post-process the answer to extract cited regions ... render them as overlays"
//
// References:
//   ColPali / late-interaction retrieval  https://arxiv.org/abs/2407.01449
//   Qwen3-VL bounding-box output spec     https://qwenlm.github.io/blog/qwen3-vl/
//   Canvas 2D rendering context (MDN)     https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D
//
// What this is: a small viewer that answers GET /document/:id with the page
// image URL + a JSON list of cited bounding boxes + extracted text spans, and
// serves an HTML page with inline JS that draws the overlays on a canvas.
//
// Run:  npx tsx code/main.ts            (starts viewer on http://127.0.0.1:0)
//       npx tsx code/main.ts --demo     (one self-check request + exit)

import { createServer, IncomingMessage, ServerResponse } from "node:http";
import { AddressInfo } from "node:net";

// -- Fixtures -----------------------------------------------------------------

type BoundingBox = {
  x: number;
  y: number;
  w: number;
  h: number;
};

type EvidenceRegion = {
  page: number;
  bbox: BoundingBox;
  text: string;
  score: number;
};

type DocumentFixture = {
  id: string;
  title: string;
  pageWidth: number;
  pageHeight: number;
  pageImageUrl: string;
  query: string;
  answer: string;
  evidence: EvidenceRegion[];
};

const FIXTURES: Record<string, DocumentFixture> = {
  "10k-acme-2025": {
    id: "10k-acme-2025",
    title: "Acme 10-K FY2025, Table 4",
    pageWidth: 1224,
    pageHeight: 1584,
    pageImageUrl: "/static/10k-acme-2025-p88.png",
    query: "What was Acme's free cash flow in FY2025?",
    answer:
      "Free cash flow in FY2025 was $3.12B, up from $2.41B in FY2024 (Table 4, p.88).",
    evidence: [
      {
        page: 88,
        bbox: { x: 142, y: 612, w: 410, h: 36 },
        text: "Free cash flow                    3,118    2,406",
        score: 0.91,
      },
      {
        page: 88,
        bbox: { x: 142, y: 250, w: 980, h: 24 },
        text: "Table 4. Cash Flow Summary (USD millions)",
        score: 0.74,
      },
    ],
  },
  "nature-paper-2026": {
    id: "nature-paper-2026",
    title: "Nature, late-interaction retrieval, 2026",
    pageWidth: 1200,
    pageHeight: 1553,
    pageImageUrl: "/static/nature-2026-p4.png",
    query: "What is the MaxSim reduction over BM25?",
    answer:
      "MaxSim reduces ColBERT-style query latency by 4.1x vs BM25 reranking (Fig. 3, p.4).",
    evidence: [
      {
        page: 4,
        bbox: { x: 80, y: 940, w: 520, h: 200 },
        text: "Fig. 3. End-to-end retrieval latency.",
        score: 0.88,
      },
    ],
  },
};

// -- Request parsing ----------------------------------------------------------

type ParsedRoute = {
  kind: "index" | "document" | "health" | "notfound";
  documentId?: string;
};

function parseRoute(rawUrl: string): ParsedRoute {
  const url = new URL(rawUrl, "http://viewer.local");
  const path = url.pathname;
  if (path === "/" || path === "/index.html") return { kind: "index" };
  if (path === "/health") return { kind: "health" };
  const match = /^\/document\/([A-Za-z0-9_\-]+)$/.exec(path);
  if (match) return { kind: "document", documentId: match[1] };
  return { kind: "notfound" };
}

// -- HTML rendering -----------------------------------------------------------

function renderIndex(): string {
  const items = Object.values(FIXTURES)
    .map(
      (d) =>
        `<li><a href="/document/${d.id}">${d.title}</a> - <em>${d.query}</em></li>`,
    )
    .join("\n");
  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Document QA viewer</title>
<style>body{font-family:system-ui,sans-serif;max-width:720px;margin:2rem auto;color:#222}</style>
</head><body>
<h1>Capstone 04 viewer</h1>
<p>Pick a document. Cited regions render as canvas overlays on the page image.</p>
<ul>${items}</ul>
</body></html>`;
}

function renderDocument(doc: DocumentFixture): string {
  const payload = JSON.stringify({
    id: doc.id,
    pageWidth: doc.pageWidth,
    pageHeight: doc.pageHeight,
    pageImageUrl: doc.pageImageUrl,
    evidence: doc.evidence,
  });
  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>${doc.title}</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 980px; margin: 2rem auto; color: #222; }
  .stage { position: relative; border: 1px solid #ddd; display: inline-block; }
  canvas.overlay { position: absolute; top: 0; left: 0; pointer-events: none; }
  .answer { background: #f6f6f6; padding: 1rem; border-left: 4px solid #444; }
  .evidence li { margin-bottom: .5rem; }
</style></head><body>
<h1>${doc.title}</h1>
<p><strong>Q:</strong> ${doc.query}</p>
<div class="answer"><strong>A:</strong> ${doc.answer}</div>
<h2>Page (page image + overlays)</h2>
<div class="stage" id="stage" style="width:${doc.pageWidth / 2}px;height:${doc.pageHeight / 2}px;background:#fafafa">
  <canvas class="overlay" id="overlay" width="${doc.pageWidth / 2}" height="${doc.pageHeight / 2}"></canvas>
</div>
<h2>Cited regions</h2>
<ul class="evidence">
${doc.evidence
  .map(
    (e, i) =>
      `<li><strong>#${i + 1}</strong> (score ${e.score.toFixed(2)}): <code>${e.text}</code></li>`,
  )
  .join("\n")}
</ul>
<script>
  const DATA = ${payload};
  function draw() {
    const c = document.getElementById("overlay");
    const ctx = c.getContext("2d");
    if (!ctx) return;
    const sx = c.width / DATA.pageWidth;
    const sy = c.height / DATA.pageHeight;
    ctx.lineWidth = 2;
    ctx.font = "12px system-ui";
    DATA.evidence.forEach((e, i) => {
      const hue = 200 + i * 40;
      ctx.strokeStyle = "hsl(" + hue + ",70%,45%)";
      ctx.fillStyle = "hsla(" + hue + ",70%,45%,0.18)";
      const x = e.bbox.x * sx;
      const y = e.bbox.y * sy;
      const w = e.bbox.w * sx;
      const h = e.bbox.h * sy;
      ctx.fillRect(x, y, w, h);
      ctx.strokeRect(x, y, w, h);
      ctx.fillStyle = "hsl(" + hue + ",70%,30%)";
      ctx.fillText("#" + (i + 1), x + 4, y + 14);
    });
  }
  if (typeof document !== "undefined") draw();
</script>
</body></html>`;
}

// -- HTTP handler -------------------------------------------------------------

function writeText(res: ServerResponse, status: number, body: string, contentType: string): void {
  res.writeHead(status, {
    "content-type": contentType,
    "content-length": Buffer.byteLength(body),
    "cache-control": "no-store",
  });
  res.end(body);
}

function handle(req: IncomingMessage, res: ServerResponse): void {
  const route = parseRoute(req.url ?? "/");
  if (route.kind === "health") {
    writeText(res, 200, JSON.stringify({ ok: true }), "application/json");
    return;
  }
  if (route.kind === "index") {
    writeText(res, 200, renderIndex(), "text/html; charset=utf-8");
    return;
  }
  if (route.kind === "document" && route.documentId) {
    const doc = FIXTURES[route.documentId];
    if (!doc) {
      writeText(res, 404, JSON.stringify({ error: "unknown document" }), "application/json");
      return;
    }
    const accept = String(req.headers["accept"] ?? "");
    if (accept.includes("application/json")) {
      writeText(
        res,
        200,
        JSON.stringify({
          id: doc.id,
          title: doc.title,
          query: doc.query,
          answer: doc.answer,
          pageWidth: doc.pageWidth,
          pageHeight: doc.pageHeight,
          pageImageUrl: doc.pageImageUrl,
          evidence: doc.evidence,
        }),
        "application/json",
      );
      return;
    }
    writeText(res, 200, renderDocument(doc), "text/html; charset=utf-8");
    return;
  }
  writeText(res, 404, JSON.stringify({ error: "not found" }), "application/json");
}

// -- Self-check ---------------------------------------------------------------

async function runDemo(): Promise<void> {
  const server = createServer(handle);
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const port = (server.address() as AddressInfo).port;
  const base = `http://127.0.0.1:${port}`;

  const probes: Array<{ label: string; path: string; accept?: string }> = [
    { label: "GET /health", path: "/health", accept: "application/json" },
    { label: "GET / (index html)", path: "/", accept: "text/html" },
    {
      label: "GET /document/10k-acme-2025 (json)",
      path: "/document/10k-acme-2025",
      accept: "application/json",
    },
    {
      label: "GET /document/10k-acme-2025 (html)",
      path: "/document/10k-acme-2025",
      accept: "text/html",
    },
    {
      label: "GET /document/nature-paper-2026 (json)",
      path: "/document/nature-paper-2026",
      accept: "application/json",
    },
    {
      label: "GET /document/missing (404)",
      path: "/document/missing",
      accept: "application/json",
    },
  ];

  console.log("=".repeat(72));
  console.log("CAPSTONE 04 - DOCUMENT QA VIEWER SKELETON (TypeScript)");
  console.log("=".repeat(72));

  let ok = 0;
  for (const probe of probes) {
    const resp = await fetch(`${base}${probe.path}`, {
      headers: probe.accept ? { accept: probe.accept } : undefined,
    });
    const body = await resp.text();
    const preview = body.replace(/\s+/g, " ").slice(0, 80);
    console.log(`\n${probe.label}`);
    console.log(`  status=${resp.status} ct=${resp.headers.get("content-type") ?? ""}`);
    console.log(`  body[:80]=${preview}`);
    const expected = probe.path === "/document/missing" ? 404 : 200;
    if (resp.status === expected) ok += 1;
  }

  console.log("\n" + "-".repeat(72));
  console.log(`probes ok=${ok}/${probes.length}`);
  await new Promise<void>((resolve) => server.close(() => resolve()));
}

function main(): void {
  if (process.argv.includes("--demo") || !process.stdout.isTTY) {
    runDemo()
      .then(() => process.exit(0))
      .catch((err: unknown) => {
        console.error("demo failed:", err);
        process.exit(1);
      });
    return;
  }
  const server = createServer(handle);
  const port = Number(process.env.PORT ?? 0);
  server.listen(port, "127.0.0.1", () => {
    const addr = server.address() as AddressInfo;
    console.log(`viewer listening on http://127.0.0.1:${addr.port}`);
  });
}

main();
