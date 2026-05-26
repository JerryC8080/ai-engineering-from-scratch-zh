// Capstone 08 - Production RAG chatbot: chat UI skeleton (TypeScript stdlib).
//
// Source: phases/19-capstone-projects/08-production-rag-chatbot/docs/en.md
//   "Languages: Python (pipeline + API), TypeScript (chat UI)"
//   "emits a citation-anchored response"
//
// References:
//   Server-Sent Events (WHATWG)        https://html.spec.whatwg.org/multipage/server-sent-events.html
//   text/event-stream (RFC 8895 / MDN) https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events
//   EventSource interface (MDN)        https://developer.mozilla.org/en-US/docs/Web/API/EventSource
//
// Stdlib only. SSE /chat/stream emits mocked token chunks. Conversation state
// lives in a Map keyed by sessionId. Minimal HTML/JS client included.
//
// Run:  npx tsx code/main.ts            (HTTP server on 127.0.0.1:0)
//       npx tsx code/main.ts --demo     (self-check, exits 0)

import { createServer, IncomingMessage, ServerResponse } from "node:http";
import { AddressInfo } from "node:net";
import { randomUUID } from "node:crypto";

// -- Conversation state -------------------------------------------------------

type Turn = { role: "user" | "assistant"; content: string; ts: number };
type Session = {
  id: string;
  role: string;
  jurisdiction: string;
  turns: Turn[];
  createdAt: number;
};

const SESSIONS = new Map<string, Session>();

function getOrCreateSession(id: string, role: string, jurisdiction: string): Session {
  const existing = SESSIONS.get(id);
  if (existing) return existing;
  const session: Session = {
    id,
    role,
    jurisdiction,
    turns: [],
    createdAt: Date.now(),
  };
  SESSIONS.set(id, session);
  return session;
}

// -- Mock retrieval + answer ---------------------------------------------------

type Citation = {
  docId: string;
  page: number;
  snippet: string;
  score: number;
};

const KB: Array<{ docId: string; page: number; text: string; tag: string }> = [
  {
    docId: "GDPR-Art-15",
    page: 1,
    text: "The data subject has the right to obtain confirmation as to whether personal data are being processed.",
    tag: "GDPR",
  },
  {
    docId: "GDPR-Art-17",
    page: 1,
    text: "The data subject shall have the right to obtain erasure of personal data without undue delay.",
    tag: "GDPR",
  },
  {
    docId: "HIPAA-164.502",
    page: 14,
    text: "Covered entity may not use or disclose protected health information except as permitted.",
    tag: "HIPAA",
  },
  {
    docId: "SOC2-CC6.1",
    page: 7,
    text: "Logical access controls restrict access to information assets to authorized users.",
    tag: "SOC2",
  },
];

function retrieve(query: string, jurisdiction: string, k: number): Citation[] {
  const tokens = new Set(query.toLowerCase().split(/\W+/).filter(Boolean));
  const scored = KB.map((doc) => {
    const docTokens = doc.text.toLowerCase().split(/\W+/);
    let overlap = 0;
    for (const t of docTokens) if (tokens.has(t)) overlap += 1;
    const boost = doc.tag === jurisdiction ? 2 : 0;
    return {
      citation: {
        docId: doc.docId,
        page: doc.page,
        snippet: doc.text,
        score: overlap + boost,
      },
      score: overlap + boost,
    };
  });
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, k).map((s) => s.citation);
}

function tokenizeAnswer(query: string, citations: Citation[]): string[] {
  const lead =
    citations.length === 0
      ? `No matching policy found for "${query}".`
      : `Per ${citations[0].docId}, ${citations[0].snippet}`;
  const tail = citations.length > 1
    ? ` See also ${citations.slice(1).map((c) => c.docId).join(", ")}.`
    : "";
  return (lead + tail).split(/(\s+)/).filter((t) => t.length > 0);
}

// -- SSE writer ---------------------------------------------------------------

function writeSseFrame(res: ServerResponse, event: string, data: unknown): void {
  res.write(`event: ${event}\n`);
  res.write(`data: ${JSON.stringify(data)}\n\n`);
}

async function sleep(ms: number): Promise<void> {
  await new Promise<void>((resolve) => setTimeout(resolve, ms));
}

// -- HTML client --------------------------------------------------------------

function renderClient(): string {
  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Production RAG chatbot</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 720px; margin: 2rem auto; color: #222; }
  #log { border: 1px solid #ddd; padding: 1rem; min-height: 200px; white-space: pre-wrap; }
  .role { color: #666; font-size: .85rem; }
  form { margin-top: 1rem; display: flex; gap: .5rem; }
  input[type=text] { flex: 1; padding: .5rem; }
  .cites { margin-top: 1rem; font-size: .9rem; color: #333; }
</style></head><body>
<h1>Capstone 08 chat (skeleton)</h1>
<p>Role: <code>analyst</code>, jurisdiction: <code>GDPR</code>. Streams SSE token-by-token.</p>
<div id="log"></div>
<div class="cites" id="cites"></div>
<form id="f">
  <input type="text" id="q" placeholder="ask about a policy..." required>
  <button type="submit">send</button>
</form>
<script>
  const sessionId = "demo-session";
  const role = "analyst";
  const jurisdiction = "GDPR";
  const log = document.getElementById("log");
  const cites = document.getElementById("cites");
  document.getElementById("f").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const q = document.getElementById("q").value;
    log.textContent += "\\nuser: " + q + "\\nassistant: ";
    cites.textContent = "";
    const url = "/chat/stream?sessionId=" + encodeURIComponent(sessionId)
      + "&role=" + encodeURIComponent(role)
      + "&jurisdiction=" + encodeURIComponent(jurisdiction)
      + "&q=" + encodeURIComponent(q);
    const es = new EventSource(url);
    es.addEventListener("token", (e) => {
      const data = JSON.parse(e.data);
      log.textContent += data.text;
    });
    es.addEventListener("citations", (e) => {
      const data = JSON.parse(e.data);
      cites.textContent = "citations: " + data.items.map((c) => c.docId + " p." + c.page).join(", ");
    });
    es.addEventListener("done", () => { es.close(); });
    es.onerror = () => { es.close(); };
  });
</script></body></html>`;
}

// -- HTTP handler -------------------------------------------------------------

function writeJson(res: ServerResponse, status: number, body: unknown): void {
  const payload = JSON.stringify(body);
  res.writeHead(status, {
    "content-type": "application/json",
    "content-length": Buffer.byteLength(payload),
  });
  res.end(payload);
}

function writeHtml(res: ServerResponse, status: number, body: string): void {
  res.writeHead(status, {
    "content-type": "text/html; charset=utf-8",
    "content-length": Buffer.byteLength(body),
  });
  res.end(body);
}

async function handleChatStream(
  req: IncomingMessage,
  res: ServerResponse,
  url: URL,
): Promise<void> {
  const sessionId = url.searchParams.get("sessionId") ?? randomUUID();
  const role = url.searchParams.get("role") ?? "analyst";
  const jurisdiction = url.searchParams.get("jurisdiction") ?? "GDPR";
  const q = url.searchParams.get("q") ?? "";
  if (!q) {
    writeJson(res, 400, { error: "missing q" });
    return;
  }

  const session = getOrCreateSession(sessionId, role, jurisdiction);
  session.turns.push({ role: "user", content: q, ts: Date.now() });

  res.writeHead(200, {
    "content-type": "text/event-stream; charset=utf-8",
    "cache-control": "no-cache, no-transform",
    connection: "keep-alive",
    "x-accel-buffering": "no",
  });

  writeSseFrame(res, "session", { sessionId, role, jurisdiction, turn: session.turns.length });

  const citations = retrieve(q, jurisdiction, 3);
  writeSseFrame(res, "citations", { items: citations });

  const tokens = tokenizeAnswer(q, citations);
  let assembled = "";
  let aborted = false;
  req.on("close", () => {
    aborted = true;
  });
  for (const tok of tokens) {
    if (aborted) return;
    assembled += tok;
    writeSseFrame(res, "token", { text: tok });
    await sleep(5);
  }
  session.turns.push({ role: "assistant", content: assembled, ts: Date.now() });
  writeSseFrame(res, "done", { totalTokens: tokens.length });
  res.end();
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? "/", "http://chat.local");
  if (req.method === "GET" && url.pathname === "/") {
    writeHtml(res, 200, renderClient());
    return;
  }
  if (req.method === "GET" && url.pathname === "/health") {
    writeJson(res, 200, { ok: true, sessions: SESSIONS.size });
    return;
  }
  if (req.method === "GET" && url.pathname === "/chat/stream") {
    await handleChatStream(req, res, url);
    return;
  }
  if (req.method === "GET" && url.pathname === "/sessions") {
    const list = Array.from(SESSIONS.values()).map((s) => ({
      id: s.id,
      role: s.role,
      jurisdiction: s.jurisdiction,
      turnCount: s.turns.length,
    }));
    writeJson(res, 200, { sessions: list });
    return;
  }
  writeJson(res, 404, { error: "not found" });
}

// -- Self-check ---------------------------------------------------------------

async function consumeSse(url: string): Promise<{ events: Array<{ event: string; data: unknown }> }> {
  const resp = await fetch(url, { headers: { accept: "text/event-stream" } });
  if (!resp.body) return { events: [] };
  const decoder = new TextDecoder();
  let buffer = "";
  const events: Array<{ event: string; data: unknown }> = [];
  const reader = resp.body.getReader();
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx = buffer.indexOf("\n\n");
    while (idx !== -1) {
      const frame = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      idx = buffer.indexOf("\n\n");
      let eventName = "message";
      const dataLines: string[] = [];
      for (const line of frame.split("\n")) {
        if (line.startsWith("event: ")) eventName = line.slice(7);
        else if (line.startsWith("data: ")) dataLines.push(line.slice(6));
      }
      if (dataLines.length === 0) continue;
      try {
        events.push({ event: eventName, data: JSON.parse(dataLines.join("\n")) });
      } catch {
        events.push({ event: eventName, data: dataLines.join("\n") });
      }
    }
  }
  return { events };
}

async function runDemo(): Promise<void> {
  const server = createServer((req, res) => {
    handle(req, res).catch((err: unknown) => {
      writeJson(res, 500, { error: String(err) });
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const port = (server.address() as AddressInfo).port;
  const base = `http://127.0.0.1:${port}`;

  console.log("=".repeat(72));
  console.log("CAPSTONE 08 - PRODUCTION RAG CHAT UI SKELETON (TypeScript)");
  console.log("=".repeat(72));

  console.log("\nGET /");
  const indexResp = await fetch(`${base}/`);
  console.log(`  status=${indexResp.status} ct=${indexResp.headers.get("content-type")}`);

  console.log("\nGET /chat/stream (q=erasure right)");
  const stream1 = await consumeSse(
    `${base}/chat/stream?sessionId=s-1&role=analyst&jurisdiction=GDPR&q=erasure%20right`,
  );
  console.log(`  events=${stream1.events.length}`);
  const tokenCount = stream1.events.filter((e) => e.event === "token").length;
  const citEvent = stream1.events.find((e) => e.event === "citations");
  console.log(`  token events=${tokenCount}`);
  console.log(`  citations event=${JSON.stringify(citEvent?.data).slice(0, 140)}`);
  console.log(`  has done=${stream1.events.some((e) => e.event === "done")}`);

  console.log("\nGET /chat/stream (same session, second turn)");
  const stream2 = await consumeSse(
    `${base}/chat/stream?sessionId=s-1&role=analyst&jurisdiction=GDPR&q=access%20confirmation`,
  );
  console.log(`  events=${stream2.events.length}`);

  console.log("\nGET /sessions");
  const sessResp = await fetch(`${base}/sessions`);
  const sessJson = (await sessResp.json()) as { sessions: Array<{ id: string; turnCount: number }> };
  console.log(`  sessions=${sessJson.sessions.length}`);
  console.log(`  s-1 turns=${sessJson.sessions.find((s) => s.id === "s-1")?.turnCount ?? 0}`);

  console.log("\nGET /chat/stream missing q (400)");
  const badResp = await fetch(`${base}/chat/stream`);
  console.log(`  status=${badResp.status}`);

  const expected =
    indexResp.status === 200 &&
    tokenCount > 0 &&
    stream1.events.some((e) => e.event === "done") &&
    badResp.status === 400 &&
    (sessJson.sessions.find((s) => s.id === "s-1")?.turnCount ?? 0) === 4;

  console.log("\n" + "-".repeat(72));
  console.log(`smoke ok=${expected}`);
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
  const server = createServer((req, res) => {
    handle(req, res).catch((err: unknown) => {
      writeJson(res, 500, { error: String(err) });
    });
  });
  const port = Number(process.env.PORT ?? 0);
  server.listen(port, "127.0.0.1", () => {
    const addr = server.address() as AddressInfo;
    console.log(`chat-ui listening on http://127.0.0.1:${addr.port}`);
  });
}

main();
