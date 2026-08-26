# PV Registration Lab

This directory is a self-contained code bundle for private multi-station PV
registration experiments with TabM. Only datasets, the station-capacity CSV,
and installed Python packages are external.

## Included code

- `pipelines/tabm4pv.py`: original TabM training, inference, embeddings, and
  scoring reference.
- `pipelines/registered_tabm4pv.py`: previous combined registration + TabM
  reference.
- `pipelines/pv_curve_registration.py`: dependency of the combined reference.
- `editable/`: algorithms the research agent may change.
- `adapters/project_adapter.py`: controlled bridge from an experiment request
  to the bundled pipeline logic.
- `controller.py`: identity, ablation, LOSO, reporting, snapshot, and final-test
  gates.

## Private-environment setup

```bash
python -m pip install -r requirements.txt
cp config.example.json config.json
```

Set `data_contract.parquet_root`, `source_stations`, and the adapter command in
`config.json`. Implement `adapters/project_adapter.py` using only bundled Python
code. After the adapter is complete, initialize the protected protocol once:

```bash
python controller.py init
python controller.py check
python controller.py identity
python controller.py next
```

Do not copy private parquet files, `station_info.csv`, `config.json`, or
`.lab_state/` back to a public repository.
