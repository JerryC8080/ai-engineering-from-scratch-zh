import { randomUUID } from "node:crypto";
import type { Counters, GenAISpan } from "./types.js";

export class RingBuffer<T> {
  private readonly capacity: number;
  private readonly slots: (T | undefined)[];
  private writeIdx = 0;
  private filled = false;

  constructor(capacity: number) {
    if (capacity <= 0) throw new Error("capacity must be > 0");
    this.capacity = capacity;
    this.slots = new Array<T | undefined>(capacity);
  }

  push(item: T): void {
    this.slots[this.writeIdx] = item;
    this.writeIdx = (this.writeIdx + 1) % this.capacity;
    if (this.writeIdx === 0) this.filled = true;
  }

  size(): number {
    return this.filled ? this.capacity : this.writeIdx;
  }

  isFull(): boolean {
    return this.filled;
  }

  snapshot(): T[] {
    if (!this.filled) return this.slots.slice(0, this.writeIdx) as T[];
    return [
      ...(this.slots.slice(this.writeIdx) as T[]),
      ...(this.slots.slice(0, this.writeIdx) as T[]),
    ];
  }
}

export function normaliseSpan(raw: unknown): GenAISpan | null {
  if (!raw || typeof raw !== "object") return null;
  const r = raw as Record<string, unknown>;
  const attrs = (r["attributes"] ?? {}) as Record<string, unknown>;
  if (typeof attrs["gen_ai.system"] !== "string") return null;
  if (typeof attrs["gen_ai.request.model"] !== "string") return null;
  const start = Number(r["start_time_unix_nano"] ?? 0);
  const end = Number(r["end_time_unix_nano"] ?? start);
  if (!Number.isFinite(start) || !Number.isFinite(end)) return null;
  const span: GenAISpan = {
    trace_id: typeof r["trace_id"] === "string" ? r["trace_id"] : randomUUID(),
    span_id:
      typeof r["span_id"] === "string"
        ? r["span_id"]
        : randomUUID().slice(0, 16),
    name: typeof r["name"] === "string" ? r["name"] : "chat.completion",
    start_time_unix_nano: start,
    end_time_unix_nano: end,
    status: r["status"] === "ERROR" ? "ERROR" : "OK",
    attributes: attrs as GenAISpan["attributes"],
  };
  if (typeof r["parent_span_id"] === "string") {
    span.parent_span_id = r["parent_span_id"];
  }
  return span;
}

export class ObservabilityStore {
  private readonly spans: RingBuffer<GenAISpan>;
  private accepted = 0;
  private rejected = 0;

  constructor(capacity = 10_000) {
    this.spans = new RingBuffer<GenAISpan>(capacity);
  }

  ingest(raw: unknown): Counters {
    const items = Array.isArray(raw) ? raw : [raw];
    for (const item of items) {
      const span = normaliseSpan(item);
      if (!span) {
        this.rejected += 1;
        continue;
      }
      this.spans.push(span);
      this.accepted += 1;
    }
    return this.counters();
  }

  snapshot(): GenAISpan[] {
    return this.spans.snapshot();
  }

  counters(): Counters {
    return {
      accepted: this.accepted,
      rejected: this.rejected,
      held: this.spans.size(),
    };
  }
}
