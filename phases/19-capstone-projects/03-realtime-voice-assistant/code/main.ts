// Capstone 19/03: realtime voice web client skeleton (TypeScript).
//
// Sources:
//   This lesson's docs/en.md (WebRTC client + VAD + barge-in client UX)
//   RFC 6455 WebSocket protocol  https://datatracker.ietf.org/doc/html/rfc6455
//   node:net + node:http upgrade https://nodejs.org/api/http.html#event-upgrade
//   Silero VAD v5 model card     https://github.com/snakers4/silero-vad
//
// The web-client side of the capstone: a minimal node:http+node:net WebSocket
// upgrade handshake, an audio-chunk transport over WS frames, a VAD state
// machine (IDLE -> LISTENING -> THINKING -> SPEAKING with barge-in), a fake
// transcript stream that drives the state machine, and a console renderer that
// shows what a browser client would render. Zero deps, deterministic.
//
// Run: npx -y tsx@4.19.2 code/main.ts

import { createHash } from "node:crypto";
import * as http from "node:http";
import type { Socket } from "node:net";

const WS_MAGIC = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

function acceptKey(clientKey: string): string {
  return createHash("sha1").update(clientKey + WS_MAGIC).digest("base64");
}

function encodeText(text: string): Buffer {
  // Server-to-client text frame, no masking per RFC 6455 section 5.2.
  const payload = Buffer.from(text, "utf8");
  const len = payload.length;
  if (len <= 125) {
    const header = Buffer.alloc(2);
    header[0] = 0x81;
    header[1] = len;
    return Buffer.concat([header, payload]);
  }
  if (len <= 0xffff) {
    const header = Buffer.alloc(4);
    header[0] = 0x81;
    header[1] = 126;
    header.writeUInt16BE(len, 2);
    return Buffer.concat([header, payload]);
  }
  const header = Buffer.alloc(10);
  header[0] = 0x81;
  header[1] = 127;
  header.writeBigUInt64BE(BigInt(len), 2);
  return Buffer.concat([header, payload]);
}

type State = "IDLE" | "LISTENING" | "WAITING" | "THINKING" | "SPEAKING";

type AudioChunk = {
  tMs: number;
  isSpeech: boolean;
  partial: string;
};

type Tool = { name: string; latencyMs: number; result: string };

const WEATHER: Tool = {
  name: "weather.tokyo_tomorrow",
  latencyMs: 420,
  result: "68/52 partly cloudy",
};

function turnCompletionScore(partial: string): number {
  // Tiny stand-in for the LiveKit turn-detector model.
  if (!partial) return 0;
  const tail = partial.trimEnd();
  if (tail.endsWith("?") || tail.endsWith(".") || tail.endsWith("!")) return 0.95;
  const n = partial.split(/\s+/).filter(Boolean).length;
  if (n < 3) return 0.2;
  if (n < 6) return 0.55;
  return 0.75;
}

function synthCall(script: string, startMs = 0, noise = 0): AudioChunk[] {
  // Generate 20ms-frame "audio" with a leading silence, then per-word speech,
  // then a long trailing silence so the state machine can run end to end.
  const words = script.split(" ");
  const frames: AudioChunk[] = [];
  let t = startMs;
  for (let i = 0; i < 6; i++) {
    frames.push({ tMs: t, isSpeech: Math.random() < noise, partial: "" });
    t += 20;
  }
  let partial = "";
  for (const w of words) {
    partial = (partial ? partial + " " : "") + w;
    for (let i = 0; i < 16; i++) {
      frames.push({ tMs: t, isSpeech: true, partial });
      t += 20;
    }
  }
  for (let i = 0; i < 110; i++) {
    frames.push({ tMs: t, isSpeech: false, partial });
    t += 20;
  }
  return frames;
}

type Metrics = {
  events: string[];
  turnCompleteMs: number;
  firstLlmTokenMs: number;
  firstAudioOutMs: number;
  bargeIns: number;
};

function newMetrics(): Metrics {
  return {
    events: [],
    turnCompleteMs: 0,
    firstLlmTokenMs: 0,
    firstAudioOutMs: 0,
    bargeIns: 0,
  };
}

function turnLatencyMs(m: Metrics): number {
  if (m.turnCompleteMs && m.firstAudioOutMs) return m.firstAudioOutMs - m.turnCompleteMs;
  return -1;
}

type SessionOptions = {
  useTool: boolean;
  bargeInAtMs: number | null;
  onEvent?: (line: string) => void;
};

function runSession(frames: AudioChunk[], opts: SessionOptions): Metrics {
  const m = newMetrics();
  let state: State = "IDLE";
  let silenceRunMs = 0;
  let finalPartial = "";
  let llmStartedAt = -1;
  let ttsStartedAt = -1;
  let toolStartedAt = -1;
  let fillerEmitted = false;
  let toolPhase: "none" | "running" | "done" = "none";

  const log = (line: string): void => {
    m.events.push(line);
    opts.onEvent?.(line);
  };

  for (const f of frames) {
    if (
      opts.bargeInAtMs !== null &&
      f.tMs >= opts.bargeInAtMs &&
      (state === "SPEAKING" || state === "THINKING") &&
      f.isSpeech
    ) {
      m.bargeIns += 1;
      log(`${f.tMs}ms BARGE-IN: cancel TTS, re-arm ASR`);
      state = "LISTENING";
      ttsStartedAt = -1;
      llmStartedAt = -1;
      continue;
    }

    if (state === "IDLE") {
      if (f.isSpeech) {
        state = "LISTENING";
        log(`${f.tMs}ms LISTENING`);
      }
      continue;
    }

    if (state === "LISTENING") {
      if (f.isSpeech) {
        silenceRunMs = 0;
        finalPartial = f.partial || finalPartial;
      } else {
        silenceRunMs += 20;
        if (silenceRunMs >= 500) {
          const score = turnCompletionScore(finalPartial);
          if (score >= 0.6) {
            state = "WAITING";
            m.turnCompleteMs = f.tMs;
            log(
              `${f.tMs}ms TURN COMPLETE (score=${score.toFixed(2)}) partial='${finalPartial}'`,
            );
          } else {
            log(`${f.tMs}ms SILENCE but score=${score.toFixed(2)}, waiting`);
          }
        }
      }
    }

    if (state === "WAITING") {
      if (opts.useTool && toolPhase === "none") {
        toolStartedAt = f.tMs;
        toolPhase = "running";
        log(`${f.tMs}ms tool call fired: ${WEATHER.name}`);
        state = "THINKING";
      } else {
        llmStartedAt = f.tMs + 140;
        state = "THINKING";
        log(`${f.tMs}ms LLM call fired`);
      }
      continue;
    }

    if (state === "THINKING") {
      if (toolPhase === "running") {
        if (!fillerEmitted && f.tMs - toolStartedAt >= 300) {
          fillerEmitted = true;
          log(`${f.tMs}ms filler 'one second, let me check'`);
        }
        if (f.tMs - toolStartedAt >= WEATHER.latencyMs) {
          toolPhase = "done";
          log(`${f.tMs}ms tool result: ${WEATHER.result}`);
          llmStartedAt = f.tMs + 140;
        }
      } else if (llmStartedAt > 0 && f.tMs >= llmStartedAt) {
        if (m.firstLlmTokenMs === 0) {
          m.firstLlmTokenMs = f.tMs;
          log(`${f.tMs}ms LLM first token`);
        }
        ttsStartedAt = f.tMs + 180;
        state = "SPEAKING";
      }
      continue;
    }

    if (state === "SPEAKING") {
      if (ttsStartedAt > 0 && f.tMs >= ttsStartedAt && m.firstAudioOutMs === 0) {
        m.firstAudioOutMs = f.tMs;
        log(`${f.tMs}ms TTS first audio-out`);
      }
    }
  }
  return m;
}

function renderToConsole(label: string, m: Metrics): void {
  console.log(`=== ${label} ===`);
  for (const line of m.events) console.log(" ", line);
  console.log(`  turn_complete   @ ${m.turnCompleteMs}ms`);
  console.log(`  first_llm_token @ ${m.firstLlmTokenMs}ms`);
  console.log(`  first_audio_out @ ${m.firstAudioOutMs}ms`);
  console.log(`  turn latency    = ${turnLatencyMs(m)}ms`);
  console.log(`  barge_ins       = ${m.bargeIns}`);
  console.log("");
}

function handleUpgrade(req: http.IncomingMessage, socket: Socket): void {
  const key = req.headers["sec-websocket-key"];
  if (typeof key !== "string") {
    socket.end("HTTP/1.1 400 Bad Request\r\n\r\n");
    return;
  }
  const accept = acceptKey(key);
  socket.write(
    "HTTP/1.1 101 Switching Protocols\r\n" +
      "Upgrade: websocket\r\n" +
      "Connection: Upgrade\r\n" +
      `Sec-WebSocket-Accept: ${accept}\r\n\r\n`,
  );

  // Drive a scripted session over the live WS as if it were a browser client.
  const frames = synthCall("what is the weather in tokyo tomorrow");
  const m = runSession(frames, {
    useTool: true,
    bargeInAtMs: null,
    onEvent: (line) => socket.write(encodeText(JSON.stringify({ type: "event", line }))),
  });
  socket.write(
    encodeText(
      JSON.stringify({
        type: "summary",
        turnCompleteMs: m.turnCompleteMs,
        firstLlmTokenMs: m.firstLlmTokenMs,
        firstAudioOutMs: m.firstAudioOutMs,
        turnLatencyMs: turnLatencyMs(m),
      }),
    ),
  );
  socket.end();
}

function buildServer(): http.Server {
  const server = http.createServer((req, res) => {
    if (req.url === "/healthz") {
      const body = JSON.stringify({ ok: true });
      res.writeHead(200, {
        "content-type": "application/json",
        "content-length": Buffer.byteLength(body),
      });
      res.end(body);
      return;
    }
    res.writeHead(404).end();
  });
  server.on("upgrade", handleUpgrade);
  return server;
}

function readFrame(buf: Buffer): { text: string; rest: Buffer } | null {
  // Minimal text-frame parser. RFC 6455 5.2: server-to-client frames must have
  // the mask bit clear; client-to-server frames must have it set. We support
  // both because this skeleton drives traffic both directions.
  if (buf.length < 2) return null;
  const masked = (buf[1] & 0x80) !== 0;
  const len1 = buf[1] & 0x7f;
  let offset = 2;
  let payloadLen = len1;
  if (len1 === 126) {
    if (buf.length < 4) return null;
    payloadLen = buf.readUInt16BE(2);
    offset = 4;
  } else if (len1 === 127) {
    if (buf.length < 10) return null;
    payloadLen = Number(buf.readBigUInt64BE(2));
    offset = 10;
  }
  let mask: Buffer | null = null;
  if (masked) {
    if (buf.length < offset + 4) return null;
    mask = buf.subarray(offset, offset + 4);
    offset += 4;
  }
  if (buf.length < offset + payloadLen) return null;
  const slice = buf.subarray(offset, offset + payloadLen);
  let text: string;
  if (mask) {
    const payload = Buffer.alloc(payloadLen);
    for (let i = 0; i < payloadLen; i++) payload[i] = slice[i] ^ mask[i % 4];
    text = payload.toString("utf8");
  } else {
    text = slice.toString("utf8");
  }
  return { text, rest: buf.subarray(offset + payloadLen) };
}

async function probeWs(port: number): Promise<void> {
  // Confirm the WS upgrade by connecting as a raw client and reading events.
  const { connect } = await import("node:net");
  const key = Buffer.from("aie-capstone-19-03-probe").toString("base64");
  const sock = connect(port, "127.0.0.1");
  await new Promise<void>((resolve) => sock.once("connect", () => resolve()));
  sock.write(
    "GET / HTTP/1.1\r\n" +
      "Host: 127.0.0.1\r\n" +
      "Upgrade: websocket\r\n" +
      "Connection: Upgrade\r\n" +
      `Sec-WebSocket-Key: ${key}\r\n` +
      "Sec-WebSocket-Version: 13\r\n\r\n",
  );
  let buf = Buffer.alloc(0);
  let seenSummary = false;
  const events: string[] = [];
  const done = new Promise<void>((resolve) => {
    sock.on("data", (chunk: Buffer) => {
      buf = Buffer.concat([buf, chunk]);
      const headerEnd = buf.indexOf("\r\n\r\n");
      if (headerEnd >= 0) buf = buf.subarray(headerEnd + 4);
      while (true) {
        const frame = readFrame(buf);
        if (!frame) break;
        buf = frame.rest;
        try {
          const parsed = JSON.parse(frame.text) as { type?: string; line?: string };
          if (parsed.type === "event" && typeof parsed.line === "string") {
            events.push(parsed.line);
          } else if (parsed.type === "summary") {
            seenSummary = true;
            console.log(`[ws probe] frames received: ${events.length + 1}`);
            console.log(`[ws probe] summary: ${frame.text}`);
          }
        } catch {
          // Ignore malformed frames in this skeleton.
        }
      }
    });
    sock.on("end", () => resolve());
    sock.on("close", () => resolve());
  });
  await done;
  if (!seenSummary) console.log("[ws probe] did not receive summary frame");
}

async function main(): Promise<void> {
  // Pre-flight: drive two offline sessions through the state machine.
  const clean = runSession(synthCall("what is the weather in tokyo tomorrow"), {
    useTool: true,
    bargeInAtMs: null,
  });
  renderToConsole("session 1: clean call with tool (weather)", clean);

  const bargeFrames = synthCall("tell me a long story about");
  // Inject late-arriving speech to trigger barge-in.
  for (let i = 0; i < 8; i++) {
    const idx = bargeFrames.length - 20 + i;
    if (idx >= 0 && idx < bargeFrames.length) {
      bargeFrames[idx] = {
        tMs: bargeFrames[idx].tMs,
        isSpeech: true,
        partial: bargeFrames[idx].partial,
      };
    }
  }
  const bargeIn = runSession(bargeFrames, {
    useTool: false,
    bargeInAtMs: bargeFrames[bargeFrames.length - 20].tMs - 60,
  });
  renderToConsole("session 2: user barges in mid-response", bargeIn);

  // Live: stand up the WS server, drive one session over it, and tear down.
  const server = buildServer();
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", () => resolve()));
  const addr = server.address();
  if (!addr || typeof addr === "string") throw new Error("address unavailable");
  console.log(`voice-client skeleton ws://127.0.0.1:${addr.port}`);
  if (process.argv.includes("--serve")) {
    process.on("SIGINT", () => server.close(() => process.exit(0)));
    return;
  }
  await probeWs(addr.port);
  await new Promise<void>((resolve) => server.close(() => resolve()));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
