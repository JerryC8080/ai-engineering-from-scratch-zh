---
name: skill-classification-diagnostics
description: Given a confusion matrix and class names, surface per-class failures and propose the single most impactful fix
version: 1.0.0
phase: 4
lesson: 4
tags: [computer-vision, classification, evaluation, debugging]
---

# Classification Diagnostics

A reading lens for confusion matrices. Aggregate accuracy tells you a classifier works. The confusion matrix tells you *what it does not know yet*.

## When to use

- First look at a trained classifier's validation performance.
- Between training runs to decide what to change next.
- Before shipping a model: verifying that no critical class is failing silently.
- Debugging a production regression where overall accuracy dropped one point and you need to know why.

## Inputs

- `cm`: CxC confusion matrix (rows = true, cols = predicted).
- `labels`: list of C class names, in the same order.
- Optional `class_priors`: per-class training frequency (defaults to the row sums of `cm`).

## Steps

1. **Compute per-class metrics.**
   - precision_i = cm[i,i] / sum(cm[:, i])
   - recall_i    = cm[i,i] / sum(cm[i, :])
   - f1_i        = 2 * p * r / (p + r)

2. **Rank the three worst classes** by F1. These are the candidates for targeted fixes.

3. **Find the top off-diagonal cell per row** — the one class that most commonly steals from this class. Report as `true -> predicted`.

4. **Classify the failure mode** for each worst class:
   - `ambiguity` — bidirectional confusion between two classes (cm[i,j] and cm[j,i] both high). Two classes may be genuinely similar.
   - `imbalance` — the class has much fewer examples than its confuser; the model is biased toward the majority.
   - `label_noise` — very high precision but low recall, or vice versa, suggesting mislabelled examples.
   - `systematic` — predictions spread across many other classes with no single dominant confuser; the class is poorly represented in feature space.

5. **Recommend the single most impactful next action**:
   - `ambiguity` -> collect or synthesise discriminative examples, add targeted augmentation that preserves the distinguishing feature.
   - `imbalance` -> oversample the minority class or apply class-weighted loss.
   - `label_noise` -> audit a stratified sample of the class; fix mislabels before any other change.
   - `systematic` -> increase data for the class or fine-tune with a higher weight on this class's loss.

## Report

```
[diagnostics]
  aggregate accuracy: X.XX
  macro F1:           X.XX

[top-3 worst classes]
  1. class <name>  F1 = X.XX  prec = X.XX  rec = X.XX
     top confusion: <name> -> <other>  (N cases)
     failure mode:  ambiguity | imbalance | label_noise | systematic
     action:        <one sentence>

  2. ...
  3. ...

[recommendation]
  single biggest lever: <one sentence naming the class and the fix>
```

## Rules

- Return at most three classes. More hides the signal.
- Name the dominant confuser for each worst class; never summarise as "confuses with many".
- Ground every recommendation in the confusion matrix evidence. No generic "add more data" without specifying which class.
- When precision and recall disagree by more than 0.2, always flag label noise as a candidate — real classes usually have aligned P and R after training.
