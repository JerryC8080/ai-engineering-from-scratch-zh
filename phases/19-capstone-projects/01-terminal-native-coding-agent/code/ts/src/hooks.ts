import type { HookEvent, HookFn, HookPayload, ToolArgs } from "./types.ts";

export class HookBus {
  static readonly EVENTS: HookEvent[] = [
    "SessionStart",
    "SessionEnd",
    "PreToolUse",
    "PostToolUse",
    "UserPromptSubmit",
    "Notification",
    "Stop",
    "PreCompact",
  ];

  private hooks: Map<HookEvent, HookFn[]> = new Map();

  constructor() {
    for (const e of HookBus.EVENTS) this.hooks.set(e, []);
  }

  on(event: HookEvent, fn: HookFn): void {
    this.hooks.get(event)!.push(fn);
  }

  fire(event: HookEvent, payload: HookPayload): HookPayload {
    let current = payload;
    for (const fn of this.hooks.get(event)!) {
      current = fn(current) ?? current;
    }
    return current;
  }
}

export function destructiveGuard(payload: HookPayload): HookPayload {
  const args = (payload.args ?? {}) as ToolArgs;
  const cmd = args.cmd ?? "";
  if (cmd.includes("rm -rf") || cmd.includes("shutdown")) {
    return {
      ...payload,
      blocked: true,
      reason: "destructive command blocked by PreToolUse hook",
    };
  }
  return payload;
}
