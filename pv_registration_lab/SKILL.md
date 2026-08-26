---
name: pv-registration-research
description: Diagnose and improve multi-station PV curve registration against TabM pseudo-OOD accuracy inside a private experiment environment. Use for baseline parity, time-alignment audits, ablations, source-station LOSO, evidence-driven registration changes, and sealed final-target evaluation; do not use for ordinary TabM training without registration research.
---

# PV Registration Research

Optimize downstream pseudo-OOD TabM score, not curve-alignment RMSE. Keep raw
data and private code inside the experiment environment.

## Start

1. Read [references/adapter-contract.md](references/adapter-contract.md) when
   connecting this package to the private TabM pipeline.
   The original model and combined registration references are bundled in
   `pipelines/`; Python code outside this lab is not required.
2. Copy `config.example.json` to `config.json`, configure local stations and the
   adapter command, then run:

   ```bash
   python controller.py init
   python controller.py check
   python controller.py next
   ```

3. Finish the private adapter and config before `init`. Initialization locks
   both. Never re-run `init` merely to bless an unexpected protected-file or
   configuration change.

## Research loop

Follow [references/experiment-protocol.md](references/experiment-protocol.md).
Before modifying code, produce observations, competing explanations, a
falsifiable hypothesis, a minimal experiment, and an expected result.

- Establish `baseline ~= identity` before optimizing registration.
- If parity fails, inspect the private adapter and pipeline; do not tune warps.
- Modify only `editable/`, the private adapter, new diagnostics, and new tests.
- Do not modify `controller.py`, `pvreglab/`, `protected/`, split rules, score
  definitions, acceptance thresholds, or recorded experiments.
- Run a targeted candidate first, quick ablation second, and full LOSO only for
  candidates supported by evidence.
- Judge candidates by paired TabM score across pseudo-target stations and
  seeds. Use alignment, shift, slope, round-trip, and horizon metrics only to
  explain behavior.
- Keep physical target power and target weather at the requested physical
  timestamp. If history uses canonical time, explicitly provide canonical
  current time, target time, and canonical horizon.
- Do not run `final-test` without explicit human approval.
- Use `python controller.py feedback-export` for external discussion; do not
  send local reports, raw logs, request JSON, or experiment state directly.
- Every nonbaseline request is bound to the exact content hash of `editable/`.
  A snapshot is created automatically before execution, so results from two
  algorithm versions cannot be silently pooled.
- Inspect snapshots with `python controller.py implementations`. Restore one
  only after a human selects its hash, using `python controller.py rollback
  --implementation HASH --confirm RESTORE_EDITABLE`; rerun `check` afterward.
- For `history_warp`, provide the actual timestamp of the last retained power
  point and retain interpolation margin beyond 96 points. Treat the rolling
  history as a timestamped series, not as a standalone median daily curve.

## Autonomy

The agent may invent diagnostics and registration algorithms; it is not limited
to fixed parameter search. Use low-cost tests to earn expensive experiments.
If reasoning quality is weak, fall back to predefined modes without changing
the protected protocol. Stop at configured iteration, failure, or compute
budgets.

Read [references/privacy-and-boundaries.md](references/privacy-and-boundaries.md)
before changing logging, result payloads, network behavior, or final-test flow.
