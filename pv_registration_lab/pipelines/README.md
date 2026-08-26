# Bundled forecasting pipelines

This directory keeps the model and registration code needed to connect the
private adapter without reading Python files outside `pv_registration_lab`.

- `tabm4pv.py` is the original TabM baseline reference. Preserve its model,
  numerical embedding, optimizer, label transformation, clipping, and score
  semantics when implementing `baseline` mode.
- `registered_tabm4pv.py` is the previous combined TabM/registration script.
  It is a reference for feature construction and model training, not the
  protected experiment protocol.
- `pv_curve_registration.py` is the registration dependency imported by the
  combined script.

The scripts retain their original fixed constants intentionally. Do not run
them blindly against the private target. The controlled entrypoint remains
`adapters/project_adapter.py`, which must translate an experiment request into
the corresponding data split and call equivalent pipeline logic. In controlled
experiments, weather names and the shared target/weather array index come from
`config.json`; the `FU_COV_COLUMNS` and `TARGET_INDEX` constants here are only
historical references.

Raw parquet files and `station_info.csv` remain external data inputs configured
in `config.json`; no private data belongs in this directory.
