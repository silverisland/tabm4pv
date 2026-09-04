# Evidence-Driven Experiment Protocol

## 1. Pipeline parity

Run `python controller.py identity`. If the paired score difference exceeds the
configured tolerance, stop registration research. Compare row identifiers or
aggregate row hashes, target index, weather index, capacity normalization,
splits, seeds, embeddings, preprocessing, clipping, and evaluation.

## 2. Root-cause ablation

Run `python controller.py ablation`. Interpret patterns rather than only ranks:

- Identity below baseline: pipeline mismatch.
- Phase-only stable but history-warp worse: resampling destroys useful lags.
- Winter gain and summer loss: fixed mapping confounds season.
- Boundary shifts or slopes: warp is fitting shape/amplitude artifacts.
- Gain only on a minority of stations: add confidence gating or station fallback.
- Error correlated with canonical horizon: forecast-time semantics are incomplete.

Add diagnostics when current aggregates cannot distinguish competing causes.

## 3. Hypothesis record

Before a code change, record:

```yaml
observations:
  - paired evidence
competing_explanations:
  - explanation A
  - explanation B
hypothesis: one falsifiable statement
change_scope:
  - editable/registration.py
minimal_test: one targeted held-out station and seed
expected_result: measurable direction and threshold
risk: likely failure mode
```

Change one conceptual factor per iteration. Add an invariant test for a new
mapping or time convention.

## 4. Escalation funnel

Use this order:

1. Unit and synthetic invariants.
2. One targeted pseudo-target run.
3. Quick ablation stations and seed.
4. Full source-station LOSO.
5. Multiple seeds.
6. Human-approved final target test once.

Do not promote a candidate because alignment RMSE improves. Promote it only
when paired downstream pseudo-OOD score meets acceptance criteria without an
unacceptable worst-station loss.

The source-station LOSO protocol remains genuinely station-held-out. The sealed
real-target protocol is different: it may train on the target site's 2024
history and evaluate the same physical site under its 2025 raw alias. Keep
these conclusions separate in reports.

## 5. Algorithm search order

Prefer identifiable low-degree mappings before flexible warps:

1. Identity.
2. One phase shift.
3. Seasonal phase shift.
4. Seasonal morning/noon/evening mapping.
5. Confidence-gated mapping with identity fallback.
6. Partial history mapping that protects recent lags.
7. Flexible mappings only after earlier evidence justifies them.

The agent may depart from this order when diagnostics provide concrete evidence
for another design.
