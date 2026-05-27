# AGENTS.md — Contributor and AI-Agent Operating Manual

Read this before editing anything. It is the single source of truth for repo conventions, automation contracts, and per-PR validation gates. Humans and AI agents both follow it.

The repo is a curriculum, not a SaaS app. The lessons ARE the product. Every rule below exists to keep 435 lessons coherent over time.

---

## 0. Philosophy in one paragraph

435 lessons. 20 phases. Every algorithm built from raw math before a single framework gets imported. You write backprop, the tokenizer, the attention mechanism, and the agent loop by hand in Python, TypeScript, Rust, or Julia. Then you run the same operation through the production library so the framework stops being a black box. The "Build It / Use It" split is the spine. Each lesson ships a reusable artifact you can plug into your daily workflow.

---

## 1. Repo layout

```
phases/                     # 20 phases, each contains lessons
  NN-phase-slug/
    NN-lesson-slug/
      docs/en.md            # lesson explainer (REQUIRED)
      code/                 # implementation (REQUIRED)
        main.py | main.ts | main.rs | main.jl
        tests/              # 5+ unit tests (REQUIRED if code/main.* present)
      quiz.json             # 6 questions (REQUIRED for completed lessons)
      outputs/              # reusable artifact (skill/prompt/agent/MCP server)
      mission.md            # phase-14 only — agent briefing
README.md                   # public face; auto-synced count fields
ROADMAP.md                  # phase/lesson status tracker
glossary/terms.md           # canonical term definitions
site/                       # public website (HTML + build.js)
  build.js                  # parses README + ROADMAP + glossary -> data.js
  data.js                   # GENERATED — never hand-edit
scripts/                    # automation
  audit_lessons.py          # invariant checker
  build_catalog.py          # writes catalog.json (gitignored)
  check_readme_counts.py    # hardcoded README counts vs catalog
  link_check.py             # external link health
  install_skills.py         # ships skills/ to agent home dirs
  _lib.py                   # shared frontmatter parser
.github/workflows/
  curriculum.yml            # invariant + auto-sync workflow
catalog.json                # GENERATED — gitignored, rebuilt by CI on demand
```

---

## 2. Hard rules (zero exceptions)

### 2.1 Atomic per-lesson commits
This is the repo's #1 rule. One commit per lesson directory. Never batch multiple lessons into one commit. A 12-lesson PR has 12 commits (plus optional catalog rebuild — which is now automated, so don't add one).

**Commit subject:** `<type>(phase-NN/MM): <slug or change summary>`
- Subject ≤ 72 chars (commit-validate hook enforces this)
- Body explains WHY, not WHAT
- No "Co-Authored-By: Claude" line
- No CodeRabbit / suggestions / review references in commit text
- No emoji in commit subject

**Conventional types:** `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`, `ci`, `style`, `build`, `revert`.

### 2.2 House writing style
Lesson docs follow this voice. Reviewers reject deviations.

**Banned vocabulary** (auto-rejected on PR):
`delve`, `crucial`, `robust`, `leverage`, `seamlessly`, `comprehensive`, `nuanced`, `vital`, `unlock`, `harness` (verb only — noun ok in agent-harness contexts), `bespoke`, `paradigm`, `synergy`.

**Banned punctuation:**
- Em-dashes (`—`) in body prose. Use commas, periods, or "..." Titles are the only exception.
- Smart quotes. Use straight ASCII quotes.

**Required style:**
- Cold-open sentences. No "In this lesson, you will learn..." preambles.
- Short paragraphs (2-4 sentences).
- One claim per sentence.
- Imperative voice in code comments and docstrings.

### 2.3 Diagrams
Lesson docs use **mermaid** or **SVG** only.

```
| Diagram type           | Use                                  |
|------------------------|--------------------------------------|
| State machine          | mermaid stateDiagram-v2              |
| Sequence / msg flow    | mermaid sequenceDiagram              |
| Pipeline / flow        | mermaid flowchart TD or flowchart LR |
| Class / type hierarchy | mermaid classDiagram                 |
| Architecture           | Inline SVG via assets/               |
```

**Banned:** ASCII art / Unicode box-drawing for diagrams. The bar is "would a screen reader announce this as garbage?" If yes, it's banned.

All fenced code blocks MUST have a language tag (MD040). Use `text` for plain output, `json` / `python` / `typescript` / `rust` / `julia` / `bash` / `console` / `mermaid` / `yaml` accordingly.

### 2.4 No external attribution
Lessons are ORIGINAL implementations. Do not cite, link, or attribute external repos (rasbt/LLMs-from-scratch, Karpathy/nanoGPT, FareedKhan/train-llm-from-scratch, etc.) in lesson docs, comments, or commit messages. Cite RFCs, official specs, and academic papers only when they are the canonical source for a protocol or algorithm (RFC 6455 for WebSockets, JSON Schema 2020-12 spec, MCP 2025-11-25 spec, OpenTelemetry GenAI semantic conventions, etc.). The repo is judged as an original work.

### 2.5 Dependency rules per language

| Language   | Allowed                                                                | Banned                                                                                                                                                                                  |
|------------|------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Python     | `numpy`, `torch`, `h5py`, `zstandard`, `safetensors`, stdlib           | `transformers`, `tokenizers`, `tiktoken`, `sentence-transformers`, `langchain`, `llama-index`, `chromadb`, `faiss`, `accelerate`, `deepspeed`, `fairscale`, `evaluate`, `trl`, `pyyaml` |
| TypeScript | `hono`, `zod`, `ws` (only where WS needed), `@hono/node-server`, Node 20+ stdlib | npm packages other than the allowed ones; especially banned: `@modelcontextprotocol/sdk`, `langchain`, `llamaindex`                                                                     |
| Rust       | stdlib only (single-file `rustc --edition 2021 main.rs`)               | crates.io deps (no `Cargo.toml`)                                                                                                                                                        |
| Julia      | `Random`, `Statistics`, `LinearAlgebra`, `Printf` (Julia stdlib)       | `Flux`, `Knet`, `Zygote`, `MLJ`, anything from a `Project.toml`                                                                                                                         |

If a CR suggestion or contributor asks for a banned dep, skip with reason: "stays stdlib-first for educational clarity."

### 2.6 No tracked generated files

| File                | Generated by              | State        |
|---------------------|---------------------------|--------------|
| `catalog.json`      | `scripts/build_catalog.py`| **gitignored** |
| `site/data.js`      | `node site/build.js`      | tracked, but auto-rebuilt on main push |
| `package-lock.json` | `npm install`             | **NEVER commit**, even when accidentally generated by a tool |

If a contributor includes any of the above three in a PR, request a force-clean push.

---

## 3. Lesson contract

### 3.1 docs/en.md frontmatter

Every lesson starts with this exact frontmatter:

```markdown
# <Title>

> <One-line hook — cold open, no preamble>

**Type:** <Learn | Build | Reference>
**Languages:** <Python | TypeScript | Rust | Julia | Shell | Docker — list every language with code in this lesson>
**Prerequisites:** <comma-list of upstream lesson names, or "None">
**Time:** ~<estimate in minutes>

## Learning Objectives
- <4-6 bullet points starting with a verb>

## <Free-form sections below>
```

The `**Languages:**` field MUST match the languages that have a `main.*` file in `code/`. Reviewers reject mismatches.

### 3.2 quiz.json schema (CANONICAL)

```json
{
  "lesson": "<dir-slug>",
  "title": "<Lesson Title>",
  "questions": [
    {"stage": "pre",   "question": "...", "options": ["a","b","c","d"], "correct": 0, "explanation": ""},
    {"stage": "check", "question": "...", "options": ["a","b","c","d"], "correct": 1, "explanation": ""},
    {"stage": "check", "question": "...", "options": ["a","b","c","d"], "correct": 2, "explanation": ""},
    {"stage": "check", "question": "...", "options": ["a","b","c","d"], "correct": 1, "explanation": ""},
    {"stage": "post",  "question": "...", "options": ["a","b","c","d"], "correct": 3, "explanation": ""},
    {"stage": "post",  "question": "...", "options": ["a","b","c","d"], "correct": 0, "explanation": ""}
  ]
}
```

Exactly 6 questions: **1 pre + 3 check + 2 post**. `correct` is the zero-indexed answer. Site renderer `lesson.html` only understands this shape — legacy `q/choices/answer` schemas crash silently. Always shuffle answer positions across the 6 questions (do not let `correct` cluster on index 0 or 1).

### 3.3 code/main.* requirements

- Must run end-to-end and exit 0 on `python3 code/main.py` / `npx tsx code/main.ts` / `rustc --edition 2021 main.rs && ./main` / `julia code/main.jl`.
- Self-terminating demo. No infinite stdin loops, no hanging on missing API keys.
- 4-6 line header comment at top citing the lesson's `docs/en.md` path + at most 2 spec / RFC links (no external repo links).
- 500-1000 LOC for deep capstones; 200-500 LOC for `Learn`-type lessons.

### 3.4 code/tests/

- 5+ unit tests minimum per lesson.
- Python: `python3 -m unittest discover code/tests -v` exits 0. Pytest also OK if present.
- TypeScript: `npx tsx --test tests/*.test.ts` (built-in `node:test`).
- Rust / Julia: tests live inside `main.rs` / `main.jl` as `#[test]` / `@testset` blocks.

---

## 4. Per-PR validation gates

A PR is mergeable only when ALL of the below pass. Reviewers reject PRs that skip any gate.

### 4.1 Local before pushing

Run from repo root:

```bash
# 1. Invariant checks
python3 scripts/audit_lessons.py

# 2. README count alignment (advisory locally; CI fixes on merge)
python3 scripts/check_readme_counts.py

# 3. Site catalog rebuild + sanity grep (advisory)
node site/build.js
grep -c 'tree/main/phases/' site/data.js  # should match catalog lesson count

# 4. For each lesson touched
cd phases/NN-phase/MM-lesson/code
# Python
python3 main.py && python3 -m unittest discover tests -v
# TypeScript
npx tsx main.ts && npx tsx --test tests/*.test.ts
# Rust
rustc --edition 2021 main.rs -o /tmp/main && /tmp/main
# Julia
julia main.jl
```

### 4.2 CI gates (`.github/workflows/curriculum.yml`)

| Job                              | Trigger        | Behavior                                                          |
|----------------------------------|----------------|-------------------------------------------------------------------|
| `audit`                          | push + PR      | Runs `audit_lessons.py`. **Blocking** if invariants violated.     |
| `readme-counts-sync` (main only) | push to main   | Rebuilds catalog + auto-fixes README counts + commits as bot.    |
| `site-rebuild` (main only)       | push to main   | Re-runs `node site/build.js`, commits site/data.js as bot.       |
| `readme-counts-drift`            | PR             | **Advisory only** — main self-heals on merge.                     |

### 4.3 Pull-request checklist

Copy into every PR description:

```markdown
- [ ] One commit per lesson dir (atomic-per-lesson rule)
- [ ] All affected `main.*` demos exit 0
- [ ] All affected `tests/` pass
- [ ] `python3 scripts/audit_lessons.py` clean (NN/NN lessons, 0 issues)
- [ ] docs/en.md frontmatter matches §3.1
- [ ] quiz.json matches §3.2 (1 pre + 3 check + 2 post)
- [ ] No banned vocab, no em-dashes
- [ ] All fenced code blocks language-tagged
- [ ] Diagrams are mermaid or SVG (no ASCII boxes)
- [ ] No external repo attribution in docs or comments
- [ ] No new banned deps (see §2.5)
- [ ] No package-lock.json, catalog.json, or other generated files committed
- [ ] README, ROADMAP, site/ updates left to CI (don't manually edit)
```

---

## 5. Automation contract (what CI handles vs. what you handle)

### 5.1 CI handles automatically (DO NOT touch in your PR)

| Surface              | Who updates                                  | When                          |
|----------------------|----------------------------------------------|-------------------------------|
| `catalog.json`       | nobody — gitignored, rebuilt on demand       | On every CI job that needs it |
| `README.md` counts   | `readme-counts-sync` bot                     | On push to main only          |
| `site/data.js`       | `site-rebuild` bot                           | On push to main only          |
| Bot commit author    | `github-actions[bot]`                        | Loop-guarded by message prefix |

If your PR includes a manual README-count or site-data.js change, the CI will overwrite it on merge. Don't bother editing them.

### 5.2 You handle (CI will NOT do these for you)

| Surface                                                          | When                                                       |
|------------------------------------------------------------------|------------------------------------------------------------|
| `README.md` lesson-link rows                                     | When adding a new lesson — link to `phases/NN-phase/MM-lesson/` |
| `ROADMAP.md` status                                              | When marking a lesson complete / WIP                       |
| `glossary/terms.md`                                              | When introducing a term used by more than one lesson       |
| `phases/<phase>/README.md` (if present)                          | When the phase total or new lesson lands                   |

### 5.3 Common bug: site catalog shows no entries for a phase

If `grep -c 'tree/main/phases/NN-' site/data.js` returns 0 after your PR merges, **the Phase NN rows in README are missing the `[Title](phases/NN-...)` markdown link**. `site/build.js` derives the URL from that link. Plain-text rows produce no URL. Always wrap titles in `[Title](phases/NN-phase-slug/MM-lesson-slug/)`.

---

## 6. Conflict resolution playbook

When your PR conflicts with main:

```bash
# From your branch
git fetch origin main
git merge --no-edit origin/main

# If catalog.json conflict (now impossible since gitignored, but legacy branches may still hit it):
git rm catalog.json
git commit --no-edit

# If README.md count conflict:
git checkout --theirs README.md
python3 scripts/build_catalog.py        # ephemeral
python3 scripts/check_readme_counts.py --fix
git add README.md
git commit --no-edit

# If site/data.js conflict:
git checkout --theirs site/data.js
node site/build.js
git add site/data.js
git commit --no-edit

# Push
git push origin <your-branch>
```

Never `git push --force` to a branch that has CodeRabbit comments unless you've copied the comments somewhere first — force-push detaches them.

---

## 7. CodeRabbit triage rules

CodeRabbit runs on every PR. Triage findings like this:

```
For each finding:
  if finding.category == "real bug":         FIX
  if finding.category == "security issue":   FIX
  if finding.category == "missing edge case":FIX
  if finding.category == "add tests":        SKIP (out of scope; existing tests cover surface)
  if finding.category == "add docstrings":   SKIP (terse style)
  if finding.category == "stylistic nit":    SKIP (house style)
  if finding.suggests_external_dep:          SKIP (banned, see §2.5)
  if finding.suggests_ASCII_diagram:         SKIP (banned, see §2.3 — mermaid only)
  if finding.duplicates_already_fixed:       SKIP (verify against current code first)
```

Always verify against current code before applying — CodeRabbit can flag stale references that already moved. Use `gh api repos/.../pulls/<n>/comments --paginate` to fetch findings.

Reply to CodeRabbit threads with "Fixed in <sha>" or "Skipped — <reason>" so threads close cleanly.

---

## 8. Site rebuild reference

`node site/build.js` regenerates `site/data.js` from:
- `README.md` (lesson titles, links, langs)
- `ROADMAP.md` (lesson statuses)
- `glossary/terms.md`

Any change to any of those three files invalidates `site/data.js`. CI handles the rebuild on merge to main (job `site-rebuild`). Locally, run `node site/build.js` before committing if you want the dev preview to match.

The CI job that rebuilds the site uses the `github-actions[bot]` identity. Loop prevention via commit-message prefix (`chore(site): rebuild data.js`). Rebase-and-retry up to 5 attempts on non-fast-forward.

---

## 9. New-lesson onboarding (copy-paste path)

To add lesson `MM-new-lesson` to phase `NN-phase-slug`:

```bash
# 1. Scaffold the dir
mkdir -p phases/NN-phase-slug/MM-new-lesson/{docs,code/tests,outputs}

# 2. Write docs/en.md with §3.1 frontmatter
# 3. Write code/main.<lang> with §3.3 header
# 4. Write code/tests/test_main.* with 5+ tests
# 5. Write quiz.json with §3.2 schema
# 6. (Optional) Add outputs/skill-<slug>.md if the lesson ships a skill

# 7. Update README.md to add the row:
#    | MM | [Lesson Title](phases/NN-phase-slug/MM-new-lesson/) | Type | Lang |

# 8. Update ROADMAP.md status row

# 9. Validate locally (see §4.1)

# 10. Atomic commit:
git add phases/NN-phase-slug/MM-new-lesson README.md ROADMAP.md
git commit -m "feat(phase-NN/MM): add <slug>"
# README + ROADMAP changes ride along with the lesson; site/data.js auto-regens on merge
git push -u origin <your-branch>
gh pr create --title "feat(phase-NN/MM): add <slug>" --body "<5-line summary>"
```

---

## 10. AI agent operating notes

Agents spawned to do multi-lesson work follow this contract:

1. **Worktree isolation**: each agent works in `/tmp/aie-<task-tag>/` against its own feature branch. Never edit the main repo path directly.
2. **Atomic commits**: one commit per lesson dir, even if 12 lessons land in a single PR.
3. **Self-terminating demos**: every `python3 code/main.py` / `npx tsx code/main.ts` must exit 0 within seconds. Fixture-driven, no real API calls.
4. **Mock LLMs**: scripted responses by hash of input. No `openai`, `anthropic`, `litellm` deps.
5. **Validate before push**: typecheck, run tests, run demo, run `python3 scripts/audit_lessons.py`. If any fails, fix or report — do not push broken work.
6. **PR creation**: open PR via `gh pr create` with a 5-line body. Title matches commit-validate format.
7. **Bash denial resilience**: if `Bash` denies one command, retry once with `git -C <worktree>` syntax or `Read`/`Write` tool. Do not abort on a single denial.

---

## 11. Anti-patterns to refuse

- "Refactor lesson 04 to use HuggingFace Trainer" — violates §2.5 dependency rule.
- "Add Karpathy attribution to phase 10/04" — violates §2.4 attribution rule.
- "Convert mermaid diagrams to ASCII for terminal viewers" — violates §2.3 diagram rule.
- "Combine lessons 30-33 into one mega-commit for cleaner history" — violates §2.1 atomic-per-lesson rule.
- "Add tests for every helper function" — usually SKIP per §7 (terse style; existing tests cover behavior, not implementation).
- "Use `pyyaml` for the rules engine" — banned in §2.5; use the hand-rolled YAML subset parser pattern from lesson 86.

---

## 12. Questions that should NEVER block a PR

- "Should we also update the website?" — CI handles it on merge.
- "Should we update catalog.json?" — gitignored, CI rebuilds on demand.
- "Should the README count badge be 440 or 445?" — CI fixes on merge.
- "Should we add a Co-Authored-By line?" — no, never.
- "Should we add a `Generated with Claude Code` footer?" — no, never.
- "Should we delete the lockfile we accidentally committed?" — yes, immediately. See §2.6.

If reviewers raise any of the above, point them at this file.

---

Last reviewed: 2026-05-27.
