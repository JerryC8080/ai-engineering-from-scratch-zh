import { Hono } from "hono";
import { z } from "zod";
import { buildIndex, CURRICULUM, pickNextLesson, topoOrder } from "./curriculum.js";
import type { MasteryStore } from "./mastery.js";

const SubmitBody = z.object({ correct: z.boolean() });

export function buildApp(mastery: MasteryStore): Hono {
  const app = new Hono();
  const index = buildIndex(CURRICULUM);
  const topo = topoOrder(CURRICULUM);

  app.get("/lesson/next", (c) => {
    const pick = pickNextLesson(topo, index, mastery.all(), Date.now());
    if (!pick) return c.json({ done: true, message: "curriculum complete" });
    return c.json({
      lesson: pick.lesson,
      reason: pick.reason,
      mastery: mastery.peek(pick.lesson.id) ?? null,
    });
  });

  app.post("/lesson/:id/submit", async (c) => {
    const id = c.req.param("id");
    if (!index[id]) return c.json({ error: "unknown lesson", id }, 404);
    let parsed: { correct: boolean };
    try {
      const raw = await c.req.json();
      parsed = SubmitBody.parse(raw);
    } catch (err) {
      return c.json({ error: "invalid body", detail: String(err) }, 400);
    }
    const updated = mastery.record(id, parsed.correct, Date.now());
    return c.json({ id, correct: parsed.correct, mastery: updated });
  });

  return app;
}
