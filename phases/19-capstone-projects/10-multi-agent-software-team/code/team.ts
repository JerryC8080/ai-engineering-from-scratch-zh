/**
 * Multi-Agent Software Team: agent skeleton (TypeScript).
 *
 * Mirrors the role split from docs/en.md (architect / coder / reviewer plus a
 * coordinator that ticks them in a round-robin) and the worktree-launch step
 * (Daytona sandbox per branch in production; here a child_process.execFile
 * stub that refuses denylisted shell commands). Shared workspace is in-memory.
 *
 * Source: phases/19-capstone-projects/10-multi-agent-software-team/docs/en.md
 * Stack reference: SWE-AF factory, MetaGPT roles, AutoGen 0.4 actor graph.
 *
 * Runs on Node 20+ stdlib. No npm deps. No real API calls.
 */

import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { promisify } from "node:util";

const execFileP = promisify(execFile);

type Role = "planner" | "coder" | "reviewer";

type Message = {
  from: Role | "user";
  to: Role | "broadcast";
  topic: string;
  body: string;
  ts: number;
};

type WorkspaceFile = {
  path: string;
  contents: string;
  lastWriter?: Role;
  revisions: number;
};

class SharedWorkspace {
  private readonly files = new Map<string, WorkspaceFile>();
  private readonly log: Message[] = [];

  write(path: string, contents: string, writer: Role): WorkspaceFile {
    const prev = this.files.get(path);
    const file: WorkspaceFile = {
      path,
      contents,
      lastWriter: writer,
      revisions: (prev?.revisions ?? 0) + 1,
    };
    this.files.set(path, file);
    return file;
  }

  read(path: string): WorkspaceFile | undefined {
    return this.files.get(path);
  }

  list(): WorkspaceFile[] {
    return [...this.files.values()];
  }

  fingerprint(): string {
    const hasher = createHash("sha256");
    for (const f of [...this.files.values()].sort((a, b) =>
      a.path.localeCompare(b.path),
    )) {
      hasher.update(`${f.path}:${f.contents}\n`);
    }
    return hasher.digest("hex").slice(0, 12);
  }

  appendMessage(m: Message): void {
    this.log.push(m);
  }

  messages(): readonly Message[] {
    return this.log;
  }
}

abstract class Agent {
  abstract readonly role: Role;
  protected sent = 0;
  protected received = 0;

  receive(_m: Message): void {
    this.received += 1;
  }

  protected emit(
    workspace: SharedWorkspace,
    to: Role | "broadcast",
    topic: string,
    body: string,
  ): Message {
    const message: Message = {
      from: this.role,
      to,
      topic,
      body,
      ts: Date.now(),
    };
    workspace.appendMessage(message);
    this.sent += 1;
    return message;
  }

  abstract step(workspace: SharedWorkspace, inbound: Message): Message | null;

  stats(): { role: Role; sent: number; received: number } {
    return { role: this.role, sent: this.sent, received: this.received };
  }
}

class PlannerAgent extends Agent {
  readonly role = "planner" as const;
  private planned = false;

  step(workspace: SharedWorkspace, inbound: Message): Message | null {
    super.receive(inbound);
    if (inbound.topic === "issue.opened" && !this.planned) {
      const plan = [
        "1. parse failing test in test_payments.py",
        "2. patch refund rounding in refunds.py",
        "3. add regression test test_refund_rounding",
      ].join("\n");
      workspace.write("PLAN.md", plan, this.role);
      this.planned = true;
      return this.emit(workspace, "coder", "plan.ready", plan);
    }
    if (inbound.topic === "review.changes_requested") {
      return this.emit(
        workspace,
        "coder",
        "plan.amended",
        `re-plan based on reviewer note: ${inbound.body}`,
      );
    }
    return null;
  }
}

class CoderAgent extends Agent {
  readonly role = "coder" as const;

  step(workspace: SharedWorkspace, inbound: Message): Message | null {
    super.receive(inbound);
    if (inbound.topic === "plan.ready" || inbound.topic === "plan.amended") {
      const file = workspace.read("refunds.py");
      const next =
        (file?.contents ?? "def refund(x):\n    return x\n") +
        "\n# rounding fix\n";
      workspace.write("refunds.py", next, this.role);
      workspace.write(
        "tests/test_refund_rounding.py",
        "def test_refund_rounding():\n    assert True\n",
        this.role,
      );
      return this.emit(
        workspace,
        "reviewer",
        "diff.ready",
        `fp=${workspace.fingerprint()}`,
      );
    }
    return null;
  }
}

class ReviewerAgent extends Agent {
  readonly role = "reviewer" as const;
  private reviews = 0;

  step(workspace: SharedWorkspace, inbound: Message): Message | null {
    super.receive(inbound);
    if (inbound.topic === "diff.ready") {
      this.reviews += 1;
      const plan = workspace.read("PLAN.md");
      const refunds = workspace.read("refunds.py");
      if (!plan || !refunds) {
        return this.emit(
          workspace,
          "planner",
          "review.changes_requested",
          "missing plan or refunds.py",
        );
      }
      if (this.reviews === 1) {
        return this.emit(
          workspace,
          "planner",
          "review.changes_requested",
          "test asserts True with no failure case",
        );
      }
      return this.emit(workspace, "broadcast", "review.approved", "lgtm");
    }
    return null;
  }
}

class Coordinator {
  private readonly agents: Agent[];
  private readonly workspace = new SharedWorkspace();

  constructor() {
    this.agents = [new PlannerAgent(), new CoderAgent(), new ReviewerAgent()];
  }

  private routeTo(agent: Agent, msg: Message): boolean {
    return msg.to === "broadcast" || msg.to === agent.role;
  }

  run(initialMessage: Message, maxTurns = 12): { approved: boolean; turns: number } {
    let pending: Message | null = initialMessage;
    this.workspace.appendMessage(initialMessage);
    let turn = 0;
    while (pending && turn < maxTurns) {
      let next: Message | null = null;
      for (const agent of this.agents) {
        if (!pending) break;
        if (!this.routeTo(agent, pending)) continue;
        next = agent.step(this.workspace, pending);
        if (next) break;
      }
      pending = next;
      turn += 1;
      if (pending && pending.topic === "review.approved") {
        return { approved: true, turns: turn };
      }
    }
    return { approved: false, turns: turn };
  }

  workspaceFiles(): WorkspaceFile[] {
    return this.workspace.list();
  }

  messageLog(): readonly Message[] {
    return this.workspace.messages();
  }

  stats(): { role: Role; sent: number; received: number }[] {
    return this.agents.map((a) => a.stats());
  }
}

const COMMAND_DENYLIST: ReadonlySet<string> = new Set([
  "rm",
  "sudo",
  "shutdown",
  "reboot",
  "mkfs",
  "dd",
  "curl",
  "wget",
  "chmod",
  "chown",
  "kill",
  "pkill",
]);

async function launchWorktree(args: {
  branch: string;
  command: string;
  argv: string[];
}): Promise<{ stdout: string; stderr: string; refused?: string }> {
  if (COMMAND_DENYLIST.has(args.command)) {
    return {
      stdout: "",
      stderr: "",
      refused: `command ${args.command} is denylisted in the worktree stub`,
    };
  }
  for (const arg of args.argv) {
    if (arg.includes(";") || arg.includes("&&") || arg.includes("|")) {
      return {
        stdout: "",
        stderr: "",
        refused: `arg ${arg} contains shell metacharacters`,
      };
    }
  }
  try {
    const { stdout, stderr } = await execFileP(args.command, args.argv, {
      timeout: 5_000,
      env: { ...process.env, BRANCH: args.branch },
      shell: false,
    });
    return { stdout, stderr };
  } catch (err) {
    const e = err as { stdout?: string; stderr?: string; message: string };
    return {
      stdout: e.stdout ?? "",
      stderr: e.stderr ?? e.message,
    };
  }
}

async function worktreeDemo(): Promise<void> {
  console.log("[team] worktree stub: execFile with denylist");
  const ok = await launchWorktree({
    branch: "feature/refund-rounding",
    command: "node",
    argv: ["-e", "console.log('coder sandbox ready: ' + process.env.BRANCH)"],
  });
  console.log("  node stdout:", ok.stdout.trim());
  if (ok.stderr) console.log("  node stderr:", ok.stderr.trim());

  const refused = await launchWorktree({
    branch: "feature/refund-rounding",
    command: "rm",
    argv: ["-rf", "/"],
  });
  console.log("  rm refused:", refused.refused);

  const shellInjected = await launchWorktree({
    branch: "feature/refund-rounding",
    command: "node",
    argv: ["-e", "1", ";", "echo", "pwned"],
  });
  console.log("  injection refused:", shellInjected.refused);
}

function teamDemo(): void {
  console.log("[team] coordinator demo: issue to merged diff");
  const coordinator = new Coordinator();
  const result = coordinator.run({
    from: "user",
    to: "planner",
    topic: "issue.opened",
    body: "refund amounts off-by-one cent on edge rounding cases",
    ts: Date.now(),
  });
  console.log("  approved:", result.approved, "turns:", result.turns);
  console.log("  files:");
  for (const file of coordinator.workspaceFiles()) {
    console.log(
      `    ${file.path} (writer=${file.lastWriter} rev=${file.revisions})`,
    );
  }
  console.log("  message log:");
  for (const m of coordinator.messageLog()) {
    console.log(`    ${m.from} -> ${m.to} :: ${m.topic}`);
  }
  console.log("  stats:", coordinator.stats());
}

async function main(): Promise<void> {
  teamDemo();
  console.log();
  await worktreeDemo();
}

main().catch((err) => {
  console.error("[team] fatal:", err);
  process.exit(1);
});
