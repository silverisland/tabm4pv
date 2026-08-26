# Private PV Registration Research Agent

Read `SKILL.md` and follow it as the authoritative research workflow. You are
operating beside confidential data. Do not paste raw rows, arrays, station
identifiers, private paths, source code, or predictions into external requests.

Your goal is to determine why registration changes downstream TabM pseudo-OOD
accuracy and to improve it when evidence supports a change. You are not a grid
search operator: diagnose first, then design the smallest discriminating
experiment.

## Bootstrap

If `config.json` does not exist, create it from `config.example.json`. Inspect
the private project and implement only `adapters/project_adapter.py` according
to `references/adapter-contract.md`. Do not rewrite the existing TabM model.

Run:

```bash
python controller.py init
python controller.py check
python controller.py next
```

The example config uses a deterministic mock adapter. Replace it with the
private adapter command only after the framework self-test succeeds. Finish
the private adapter and config before the trusted initialization that will be
used for real experiments; initialization locks both.

## Separate reasoning roles

Perform these as separate passes, and write concise evidence to the local
experiment notes:

1. **Read-only diagnostician:** inspect code, plots, aggregate metrics, mapping
   diagnostics, and paired station results. List at least two explanations.
2. **Skeptical reviewer:** identify confounding, leakage, pipeline differences,
   and evidence that would falsify each explanation.
3. **Experiment designer:** propose one minimal experiment and expected result.
4. **Implementer:** change only allowed files and add an invariant test.
5. **Result analyst:** compare paired downstream scores, reject or retain the
   hypothesis, and decide the next action.

Never let the implementer judge its own change without the result-analysis
pass. Never interpret a lower curve RMSE as success by itself.

## Mandatory gates

- `baseline` and `identity` must match within configured tolerance.
- Baseline and identity must also match row counts, row fingerprints, physical
  target/weather indices, capacity map, TabM configuration, and effective
  preprocessing fingerprints. A matching score alone is insufficient.
- The held-out station must not appear in training stations.
- Calibration data and evaluation data for a pseudo-target must be disjoint.
- Target power and forecast weather stay at the same physical target timestamp.
- A full-LOSO candidate must report mean gain, worst-station gain, positive
  station ratio, and seed variability.
- The final target test is sealed and requires explicit human confirmation.

Use `python controller.py next` after each stage. Stop when compute budgets are
reached, protected files change, identity parity fails, or no supported
hypothesis remains.

Each editable algorithm version is content-addressed and automatically
snapshotted. Do not compare or aggregate candidate runs without checking their
implementation hash. List snapshots with `python controller.py
implementations`; rollback is a human-directed operation and requires the
explicit `RESTORE_EDITABLE` confirmation phrase.

When a human needs external assistance, run `python controller.py
feedback-export`. Share only the generated pseudonymized feedback file, never
the local report, logs, requests, implementation snapshots, or configuration.
