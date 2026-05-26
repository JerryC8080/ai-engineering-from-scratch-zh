// Capstone 06 - DevOps troubleshooting agent: Slack-integration skeleton (TypeScript stdlib).
//
// Source: phases/19-capstone-projects/06-devops-troubleshooting-agent/docs/en.md
//   "Slack brief + approval buttons", "Destructive tools ... live on a second
//   MCP server behind an approval token. The agent can call them only after
//   the Slack card is approved by a human."
//
// References:
//   Slack request signing (v0)        https://api.slack.com/authentication/verifying-requests-from-slack
//   Slack slash commands              https://api.slack.com/interactivity/slash-commands
//   Slack Block Kit response payload  https://api.slack.com/reference/block-kit/blocks
//   HMAC-SHA256 (RFC 2104)            https://datatracker.ietf.org/doc/html/rfc2104
//
// Stdlib only. Mocks the agent + the outbound Slack response_url POST. Verifies
// request signatures with a 5-minute replay window. No real network egress.
//
// Run:  npx tsx code/main.ts            (HTTP server on 127.0.0.1:0)
//       npx tsx code/main.ts --demo     (self-check, exits 0)

import { createHmac, timingSafeEqual } from "node:crypto";
import { createServer, IncomingMessage, ServerResponse } from "node:http";
import { AddressInfo } from "node:net";

const SLACK_SIGNING_SECRET =
  process.env.SLACK_SIGNING_SECRET ?? "test-signing-secret-DO-NOT-USE-IN-PROD";
const SIGNATURE_VERSION = "v0";
const REPLAY_WINDOW_SECONDS = 60 * 5;

// -- Slack signature verification --------------------------------------------

type VerifyResult = { ok: true } | { ok: false; reason: string };

function verifySlackSignature(args: {
  signingSecret: string;
  timestamp: string;
  signature: string;
  rawBody: string;
  nowSeconds: number;
}): VerifyResult {
  const ts = Number(args.timestamp);
  if (!Number.isFinite(ts)) return { ok: false, reason: "bad-timestamp" };
  if (Math.abs(args.nowSeconds - ts) > REPLAY_WINDOW_SECONDS) {
    return { ok: false, reason: "stale" };
  }
  const base = `${SIGNATURE_VERSION}:${args.timestamp}:${args.rawBody}`;
  const computed =
    `${SIGNATURE_VERSION}=` +
    createHmac("sha256", args.signingSecret).update(base).digest("hex");
  const got = Buffer.from(args.signature);
  const want = Buffer.from(computed);
  if (got.length !== want.length) return { ok: false, reason: "length-mismatch" };
  if (!timingSafeEqual(got, want)) return { ok: false, reason: "mismatch" };
  return { ok: true };
}

function signForTesting(signingSecret: string, timestamp: string, rawBody: string): string {
  const base = `${SIGNATURE_VERSION}:${timestamp}:${rawBody}`;
  return (
    `${SIGNATURE_VERSION}=` +
    createHmac("sha256", signingSecret).update(base).digest("hex")
  );
}

// -- Mocked agent -------------------------------------------------------------

type Hypothesis = {
  rank: number;
  summary: string;
  evidence: string[];
  remediation: string;
};

type AgentReport = {
  incidentId: string;
  topHypotheses: Hypothesis[];
};

function mockAgent(alertText: string): AgentReport {
  const tokens = alertText.toLowerCase();
  if (tokens.includes("oom") || tokens.includes("memory")) {
    return {
      incidentId: `inc-${Date.now()}`,
      topHypotheses: [
        {
          rank: 1,
          summary: "Pod payments-api-7c4 OOMKilled twice in 10m, memory request 256Mi too low.",
          evidence: [
            "kube-state-metrics: kube_pod_container_status_terminated_reason{reason=OOMKilled}",
            "Prom: container_memory_working_set_bytes p99 hit limit",
          ],
          remediation: "bump payments-api request to 512Mi, limit 1Gi",
        },
        {
          rank: 2,
          summary: "Possible memory-leak introduced by v2.41 rollout (Argo).",
          evidence: ["ArgoCD: payments-api revision v2.41 deployed 14m ago"],
          remediation: "roll back payments-api to v2.40",
        },
      ],
    };
  }
  return {
    incidentId: `inc-${Date.now()}`,
    topHypotheses: [
      {
        rank: 1,
        summary: "No prior signal; agent recommends collecting telemetry.",
        evidence: ["no matching prom alerts in last 30m"],
        remediation: "no remediation proposed",
      },
    ],
  };
}

// -- Block Kit response shape -------------------------------------------------

type Block = Record<string, unknown>;

function buildSlackResponse(report: AgentReport): { response_type: string; blocks: Block[] } {
  const blocks: Block[] = [
    {
      type: "header",
      text: { type: "plain_text", text: `Incident ${report.incidentId}` },
    },
  ];
  for (const h of report.topHypotheses) {
    blocks.push({
      type: "section",
      text: {
        type: "mrkdwn",
        text: `*#${h.rank}.* ${h.summary}\nEvidence:\n- ${h.evidence.join("\n- ")}\n_Remediation:_ ${h.remediation}`,
      },
    });
  }
  blocks.push({
    type: "actions",
    elements: [
      {
        type: "button",
        text: { type: "plain_text", text: "Approve top remediation" },
        style: "primary",
        action_id: "approve",
        value: report.incidentId,
      },
      {
        type: "button",
        text: { type: "plain_text", text: "Escalate" },
        action_id: "escalate",
        value: report.incidentId,
      },
      {
        type: "button",
        text: { type: "plain_text", text: "Ignore" },
        style: "danger",
        action_id: "ignore",
        value: report.incidentId,
      },
    ],
  });
  return { response_type: "in_channel", blocks };
}

// -- Outbound Slack POST stub --------------------------------------------------

type OutboundCall = { url: string; body: unknown };
const OUTBOUND_LOG: OutboundCall[] = [];

async function postToSlackResponseUrl(url: string, body: unknown): Promise<void> {
  OUTBOUND_LOG.push({ url, body });
}

// -- Body collection -----------------------------------------------------------

async function readBody(req: IncomingMessage): Promise<string> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) {
    chunks.push(typeof chunk === "string" ? Buffer.from(chunk) : (chunk as Buffer));
  }
  return Buffer.concat(chunks).toString("utf-8");
}

// -- HTTP handler --------------------------------------------------------------

function writeJson(res: ServerResponse, status: number, body: unknown): void {
  const payload = JSON.stringify(body);
  res.writeHead(status, {
    "content-type": "application/json",
    "content-length": Buffer.byteLength(payload),
  });
  res.end(payload);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  if (req.method === "GET" && req.url === "/health") {
    writeJson(res, 200, { ok: true });
    return;
  }
  if (!(req.method === "POST" && (req.url === "/slack/command" || req.url === "/slack/interactivity"))) {
    writeJson(res, 404, { error: "not found" });
    return;
  }

  const rawBody = await readBody(req);
  const timestamp = String(req.headers["x-slack-request-timestamp"] ?? "");
  const signature = String(req.headers["x-slack-signature"] ?? "");
  const verdict = verifySlackSignature({
    signingSecret: SLACK_SIGNING_SECRET,
    timestamp,
    signature,
    rawBody,
    nowSeconds: Math.floor(Date.now() / 1000),
  });
  if (!verdict.ok) {
    writeJson(res, 401, { error: `signature ${verdict.reason}` });
    return;
  }

  if (req.url === "/slack/command") {
    const params = new URLSearchParams(rawBody);
    const text = params.get("text") ?? "";
    const responseUrl = params.get("response_url") ?? "";
    const report = mockAgent(text);
    const payload = buildSlackResponse(report);
    writeJson(res, 200, {
      response_type: "ephemeral",
      text: `Triaging incident, will follow up in <${responseUrl || "channel"}>.`,
    });
    if (responseUrl) {
      void postToSlackResponseUrl(responseUrl, payload);
    }
    return;
  }

  const params = new URLSearchParams(rawBody);
  const payloadStr = params.get("payload") ?? "{}";
  type Action = { action_id?: string; value?: string };
  type Interactivity = { actions?: Action[]; response_url?: string };
  const parsed = JSON.parse(payloadStr) as Interactivity;
  const action = parsed.actions?.[0] ?? {};
  const actionId = action.action_id ?? "unknown";
  const incidentId = action.value ?? "unknown";
  let text: string;
  if (actionId === "approve") {
    text = `Approved remediation for ${incidentId}. Calling gated MCP server (mocked).`;
  } else if (actionId === "escalate") {
    text = `Escalated ${incidentId} to on-call.`;
  } else {
    text = `Ignored ${incidentId}.`;
  }
  writeJson(res, 200, { response_type: "in_channel", replace_original: false, text });
  if (parsed.response_url) {
    void postToSlackResponseUrl(parsed.response_url, { text });
  }
}

// -- Self-check ----------------------------------------------------------------

async function postSigned(
  base: string,
  path: string,
  body: string,
  options?: { stale?: boolean; tamper?: boolean },
): Promise<Response> {
  const nowS = Math.floor(Date.now() / 1000);
  const ts = options?.stale ? String(nowS - REPLAY_WINDOW_SECONDS - 1) : String(nowS);
  let signature = signForTesting(SLACK_SIGNING_SECRET, ts, body);
  if (options?.tamper) signature = signature.slice(0, -1) + "0";
  return fetch(`${base}${path}`, {
    method: "POST",
    headers: {
      "content-type": "application/x-www-form-urlencoded",
      "x-slack-request-timestamp": ts,
      "x-slack-signature": signature,
    },
    body,
  });
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
  console.log("CAPSTONE 06 - SLACK INTEGRATION SKELETON (TypeScript)");
  console.log("=".repeat(72));

  const slashBody = new URLSearchParams({
    command: "/oncall",
    text: "OOMKilled payments-api",
    user_id: "U1",
    response_url: "https://hooks.slack.example/redacted",
  }).toString();

  const checks: Array<{ label: string; expect: number; ran: () => Promise<Response> }> = [
    {
      label: "GET /health",
      expect: 200,
      ran: () => fetch(`${base}/health`),
    },
    {
      label: "POST /slack/command with valid signature",
      expect: 200,
      ran: () => postSigned(base, "/slack/command", slashBody),
    },
    {
      label: "POST /slack/command with tampered signature",
      expect: 401,
      ran: () => postSigned(base, "/slack/command", slashBody, { tamper: true }),
    },
    {
      label: "POST /slack/command with stale timestamp",
      expect: 401,
      ran: () => postSigned(base, "/slack/command", slashBody, { stale: true }),
    },
    {
      label: "POST /slack/interactivity approve",
      expect: 200,
      ran: () =>
        postSigned(
          base,
          "/slack/interactivity",
          new URLSearchParams({
            payload: JSON.stringify({
              actions: [{ action_id: "approve", value: "inc-42" }],
              response_url: "https://hooks.slack.example/redacted",
            }),
          }).toString(),
        ),
    },
  ];

  let ok = 0;
  for (const c of checks) {
    const resp = await c.ran();
    const body = await resp.text();
    console.log(`\n${c.label}`);
    console.log(`  status=${resp.status} expect=${c.expect}`);
    console.log(`  body=${body.slice(0, 120)}`);
    if (resp.status === c.expect) ok += 1;
  }

  console.log("\n" + "-".repeat(72));
  console.log(`probes ok=${ok}/${checks.length}`);
  console.log(`outbound slack calls logged=${OUTBOUND_LOG.length}`);
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
    console.log(`slack-integration listening on http://127.0.0.1:${addr.port}`);
  });
}

main();
