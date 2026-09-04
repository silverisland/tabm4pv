# Privacy and Experiment Boundaries

- Confirm that Claude's configured model endpoint is the local DeepSeek service
  and that network egress is disabled or independently audited.
- Never place raw data, private code, file paths, station names, timestamps,
  predictions, or ground truth in model requests that leave the environment.
- Keep controller results aggregate-only. The protected evaluator rejects common
  raw-data fields but is not a substitute for operating-system and network policy.
- Treat `controller.py`, `pvreglab/`, and `protected/` as immutable. Use
  filesystem permissions in the private environment for stronger enforcement.
- Do not reset the protected manifest after an unexpected modification. Restore
  the trusted package instead.
- The real target test is not part of iterative optimization. `final-test`
  requires a human confirmation phrase and refuses a second recorded run.
- Do not change acceptance thresholds after observing candidate results unless a
  human explicitly starts a new, documented protocol version.
- Respect configured GPU time, iteration, and consecutive-failure limits.
