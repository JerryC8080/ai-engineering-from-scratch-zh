"""AI Scientist v2 loop simulator — stdlib Python.

Models the research loop as a state machine with configurable per-stage
failure probabilities, seeded from Beel et al. (2025) findings on AI
Scientist's real behavior. Runs many trials and reports the distribution
of outcomes, including the critical "polished paper with flawed
experiment" class.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


random.seed(42)


@dataclass
class LoopConfig:
    # Probability an idea is mislabeled as novel when it is not.
    novelty_mislabel: float = 0.25
    # Probability an experiment fails from coding errors (Beel et al. ~0.42).
    experiment_failure: float = 0.42
    # Fraction of experiment failures recoverable by retries.
    retry_recovery: float = 0.55
    # Probability vision-language figure critique produces clean visuals
    # even when underlying experiment is broken.
    polish_masks_weakness: float = 0.70
    # Probability the auto-writeup step produces a coherent paper given
    # (possibly flawed) experiment data.
    writeup_success: float = 0.85
    # Internal reviewer accept probability (weak reviewer).
    internal_review_accept: float = 0.50


@dataclass
class Outcome:
    submitted: bool
    has_novelty_flaw: bool
    has_experiment_flaw: bool
    polished_but_flawed: bool
    polished_ok: bool
    abandoned_stage: str


def run_one(cfg: LoopConfig) -> Outcome:
    # Idea generation always succeeds in this toy.
    has_novelty_flaw = random.random() < cfg.novelty_mislabel

    # Experiment execution: failure + retry recovery.
    failed = random.random() < cfg.experiment_failure
    if failed:
        recovered = random.random() < cfg.retry_recovery
        if not recovered:
            return Outcome(
                submitted=False,
                has_novelty_flaw=has_novelty_flaw,
                has_experiment_flaw=True,
                polished_but_flawed=False,
                polished_ok=False,
                abandoned_stage="experiment",
            )
        # Recovered, but the paper still has signs of struggle unless polished.
        has_experiment_flaw = True
    else:
        has_experiment_flaw = False

    # Vision-language figure polish.
    polished_hides_weakness = (
        has_experiment_flaw and random.random() < cfg.polish_masks_weakness
    )

    # Writeup stage.
    if random.random() > cfg.writeup_success:
        return Outcome(
            submitted=False,
            has_novelty_flaw=has_novelty_flaw,
            has_experiment_flaw=has_experiment_flaw,
            polished_but_flawed=False,
            polished_ok=False,
            abandoned_stage="writeup",
        )

    # Internal reviewer.
    if random.random() > cfg.internal_review_accept:
        return Outcome(
            submitted=False,
            has_novelty_flaw=has_novelty_flaw,
            has_experiment_flaw=has_experiment_flaw,
            polished_but_flawed=False,
            polished_ok=False,
            abandoned_stage="internal_review",
        )

    polished_ok = not has_experiment_flaw and not has_novelty_flaw
    polished_but_flawed = (
        (has_experiment_flaw and polished_hides_weakness)
        or has_novelty_flaw
    )
    return Outcome(
        submitted=True,
        has_novelty_flaw=has_novelty_flaw,
        has_experiment_flaw=has_experiment_flaw,
        polished_but_flawed=polished_but_flawed,
        polished_ok=polished_ok and not polished_but_flawed,
        abandoned_stage="",
    )


def report(n: int, cfg: LoopConfig) -> None:
    outs = [run_one(cfg) for _ in range(n)]

    submitted = [o for o in outs if o.submitted]
    abandoned = [o for o in outs if not o.submitted]
    polished_ok = [o for o in submitted if o.polished_ok]
    polished_but_flawed = [o for o in submitted if o.polished_but_flawed]

    print("  config")
    print(f"    novelty mislabel rate       : {cfg.novelty_mislabel:.2f}")
    print(f"    experiment failure rate     : {cfg.experiment_failure:.2f}")
    print(f"    retry recovery fraction     : {cfg.retry_recovery:.2f}")
    print(f"    polish masks weakness prob  : {cfg.polish_masks_weakness:.2f}")
    print(f"    writeup success rate        : {cfg.writeup_success:.2f}")
    print(f"    internal reviewer accept    : {cfg.internal_review_accept:.2f}")

    print()
    print(f"  trials                    : {n}")
    print(f"  submissions               : {len(submitted)} ({len(submitted) / n:.1%})")
    print(f"  abandoned                 : {len(abandoned)} ({len(abandoned) / n:.1%})")
    by_stage = {}
    for o in abandoned:
        by_stage[o.abandoned_stage] = by_stage.get(o.abandoned_stage, 0) + 1
    for stage, count in sorted(by_stage.items()):
        print(f"    at {stage:<18}: {count}")

    print()
    print("  submission quality breakdown")
    print(f"    clean (novel + valid)     : {len(polished_ok)} "
          f"({len(polished_ok) / n:.1%} of trials, "
          f"{len(polished_ok) / max(1, len(submitted)):.1%} of submissions)")
    print(f"    polished-but-flawed       : {len(polished_but_flawed)} "
          f"({len(polished_but_flawed) / n:.1%} of trials, "
          f"{len(polished_but_flawed) / max(1, len(submitted)):.1%} of submissions)")


def main() -> None:
    print("=" * 70)
    print("AI SCIENTIST V2 LOOP SIMULATOR (Phase 15, Lesson 5)")
    print("=" * 70)

    print("\nBaseline (Beel-style numbers)")
    print("-" * 70)
    report(1000, LoopConfig())

    print("\nOptimistic scenario (tighter numbers)")
    print("-" * 70)
    report(1000, LoopConfig(
        novelty_mislabel=0.10,
        experiment_failure=0.20,
        retry_recovery=0.80,
        polish_masks_weakness=0.40,
        writeup_success=0.92,
        internal_review_accept=0.60,
    ))

    print()
    print("=" * 70)
    print("HEADLINE: submissions outpace sound research")
    print("-" * 70)
    print("  Even in optimistic scenarios, a non-trivial share of submitted")
    print("  papers carry a flaw the polish stage helped hide. That is the")
    print("  operational meaning of 'presentation-quality gap' — and the")
    print("  reason a human review gate sits between the loop and any venue.")


if __name__ == "__main__":
    main()
