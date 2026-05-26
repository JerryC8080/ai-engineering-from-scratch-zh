import { execFile } from "node:child_process";
import { promisify } from "node:util";
import type { LaunchArgs, LaunchResult } from "./types.js";

const execFileP = promisify(execFile);

export const COMMAND_DENYLIST: ReadonlySet<string> = new Set([
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

export const SHELL_METACHARS = [";", "&&", "||", "|", "`", "$("];

export function hasShellMetachars(arg: string): boolean {
  return SHELL_METACHARS.some((m) => arg.includes(m));
}

export function refuseReason(args: LaunchArgs): string | null {
  if (COMMAND_DENYLIST.has(args.command)) {
    return `command ${args.command} is denylisted in the worktree stub`;
  }
  for (const arg of args.argv) {
    if (hasShellMetachars(arg)) {
      return `arg ${arg} contains shell metacharacters`;
    }
  }
  return null;
}

export async function launchWorktree(args: LaunchArgs): Promise<LaunchResult> {
  const refused = refuseReason(args);
  if (refused) {
    return { stdout: "", stderr: "", refused };
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
