#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pvreglab.config import load_config, resolve_paths
from pvreglab.implementation import (
    implementation_hash,
    list_implementations,
    restore_implementation,
    snapshot_implementation,
)
from pvreglab.feedback import export_feedback
from pvreglab.protection import (
    create_config_lock,
    create_manifest,
    verify_config_lock,
    verify_manifest,
)
from pvreglab.protocol import (
    ablation_requests,
    final_request,
    identity_audit_requests,
    loso_requests,
)
from pvreglab.registry import Registry
from pvreglab.report import write_report
from pvreglab.runner import ExperimentRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Protected PV registration experiment controller"
    )
    parser.add_argument("--config", default="config.json")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init", help="initialize protected hashes and state")
    subparsers.add_parser("check", help="validate config, protection, and tests")
    subparsers.add_parser("identity", help="run baseline/identity parity audit")
    subparsers.add_parser("ablation", help="run quick root-cause ablations")

    loso = subparsers.add_parser("loso", help="run full source-station LOSO")
    loso.add_argument("--modes", nargs="*")

    run = subparsers.add_parser("run", help="run one controlled candidate")
    run.add_argument("--mode", required=True)
    run.add_argument("--held-out", required=True)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--hypothesis", required=True)

    subparsers.add_parser("next", help="print the next evidence-driven action")
    subparsers.add_parser("report", help="write aggregate comparison report")
    subparsers.add_parser(
        "feedback-export", help="write a pseudonymized aggregate feedback bundle"
    )
    subparsers.add_parser("implementations", help="list editable code snapshots")

    rollback = subparsers.add_parser(
        "rollback", help="restore editable/ from a recorded snapshot"
    )
    rollback.add_argument("--implementation", required=True)
    rollback.add_argument("--confirm", required=True)

    final = subparsers.add_parser(
        "final-test", help="manually run the sealed target test once"
    )
    final.add_argument("--mode", required=True)
    final.add_argument("--seed", type=int, default=0)
    final.add_argument(
        "--confirm",
        required=True,
        help="must be exactly RUN_FINAL_TARGET_ONCE",
    )
    return parser.parse_args()


def context(args: argparse.Namespace):
    config, config_path = load_config(ROOT / args.config)
    config = resolve_paths(config, config_path)
    config["_implementation_hash"] = implementation_hash(ROOT / "editable")
    state_dir = Path(config["state_dir"])
    return config, state_dir


def init(config: dict, state_dir: Path) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = create_manifest(ROOT, state_dir, config["protected_paths"])
    config_lock = create_config_lock(Path(config["_config_path"]), state_dir)
    print(f"initialized state: {state_dir}")
    print(f"protected manifest: {path}")
    print(f"config lock: {config_lock}")


def check(config: dict, state_dir: Path) -> None:
    errors = verify_manifest(ROOT, state_dir) + verify_config_lock(
        Path(config["_config_path"]), state_dir
    )
    if errors:
        raise SystemExit("protection check failed:\n- " + "\n- ".join(errors))
    command = [str(part) for part in config["adapter_command"]]
    executable = command[0]
    if executable in {"python", "{python}"}:
        executable = sys.executable
    if not shutil.which(executable) and not Path(executable).exists():
        raise SystemExit(f"adapter executable not found: {executable}")
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=ROOT,
        check=False,
    )
    if completed.returncode:
        raise SystemExit("self-tests failed")
    print("config: ok")
    data_contract = config["data_contract"]
    print(
        "target alias: "
        f"{data_contract['target_history_raw_station']} + "
        f"{data_contract['target_evaluation_raw_station']} -> "
        f"{config['target_station']}"
    )
    print(
        f"target capacity override: "
        f"{data_contract['capacity_overrides'][config['target_station']]}"
    )
    if data_contract["allows_future_source_data"]:
        print(
            "warning: all available source data may include dates after the "
            "2025 target evaluation period; results are offline-transfer, "
            "not strict chronological backtesting"
        )
    print("protected files: ok")
    print("self-tests: ok")


def identity_audit(config: dict, records: list[dict]) -> dict | None:
    scores = {}
    audits = {}
    for record in records:
        request = record.get("request", {})
        result = record.get("result", {})
        expected = (
            config["baseline_implementation_id"]
            if request.get("mode") == "baseline"
            else config["_implementation_hash"]
        )
        if (
            request.get("stage") == "identity_audit"
            and request.get("implementation_hash") == expected
            and result.get("status") == "ok"
        ):
            scores[request.get("mode")] = float(result["score"])
            audits[request.get("mode")] = result.get("audit", {})
    if "baseline" not in scores or "identity" not in scores:
        return None
    fingerprint_keys = [
        "train_row_count",
        "validation_row_count",
        "evaluation_row_count",
        "calibration_row_count",
        "train_rows_hash",
        "validation_rows_hash",
        "evaluation_rows_hash",
        "calibration_rows_hash",
        "target_index",
        "weather_index",
        "capacity_map_hash",
        "model_config_hash",
        "preprocessing_hash",
    ]
    mismatches = [
        key
        for key in fingerprint_keys
        if audits["baseline"].get(key) != audits["identity"].get(key)
    ]
    return {
        "score_gap": scores["identity"] - scores["baseline"],
        "fingerprint_mismatches": mismatches,
    }


def next_action(config: dict, records: list[dict]) -> dict:
    audit = identity_audit(config, records)
    if audit is None:
        return {
            "action": "identity_audit",
            "command": "python controller.py identity",
            "reason": "Baseline parity has not been established.",
        }
    if (
        abs(audit["score_gap"]) > float(config["identity_score_tolerance"])
        or audit["fingerprint_mismatches"]
    ):
        return {
            "action": "diagnose_pipeline",
            "command": "Do not optimize registration. Inspect the private adapter and pipeline.",
            "evidence": {
                "identity_minus_baseline": audit["score_gap"],
                "fingerprint_mismatches": audit["fingerprint_mismatches"],
            },
            "checks": [
                "same rows and chronological split",
                "same target and weather array index",
                "same capacity normalization",
                "same TabM parameters, seed, and numerical embeddings",
                "identity history equals the original last 96 values",
            ],
        }
    stages = {
        record.get("request", {}).get("stage")
        for record in records
        if record.get("result", {}).get("status") == "ok"
    }
    if "quick_ablation" not in stages:
        return {
            "action": "quick_ablation",
            "command": "python controller.py ablation",
            "reason": "Identity parity passed; isolate the source of degradation.",
        }
    if "full_loso" not in stages:
        return {
            "action": "full_loso",
            "command": "python controller.py loso",
            "reason": "Quick evidence exists; validate candidates across pseudo-target stations.",
        }
    return {
        "action": "research_iteration",
        "command": "python controller.py report",
        "reason": (
            "Read the report and diagnostics, write a falsifiable hypothesis, "
            "change only editable/, then run a targeted candidate before LOSO."
        ),
    }


def main() -> None:
    args = parse_args()
    config, state_dir = context(args)
    if args.command == "init":
        init(config, state_dir)
        snapshot = snapshot_implementation(ROOT / "editable", state_dir)
        print(f"editable snapshot: {snapshot}")
        return
    if args.command == "check":
        check(config, state_dir)
        return

    errors = verify_manifest(ROOT, state_dir) + verify_config_lock(
        Path(config["_config_path"]), state_dir
    )
    if errors:
        raise SystemExit("protection check failed:\n- " + "\n- ".join(errors))
    runner = ExperimentRunner(ROOT, config)
    registry = Registry(state_dir)

    if args.command == "identity":
        runner.run_many(identity_audit_requests(config))
    elif args.command == "ablation":
        runner.run_many(ablation_requests(config))
    elif args.command == "loso":
        runner.run_many(loso_requests(config, args.modes))
    elif args.command == "run":
        from protected.split_protocol import make_request

        candidate_records = [
            record
            for record in registry.records()
            if record.get("request", {}).get("stage") == "agent_candidate"
        ]
        if len(candidate_records) >= int(config["budget"]["max_iterations"]):
            raise SystemExit("Agent iteration budget is exhausted")
        consecutive_failures = 0
        for record in reversed(candidate_records):
            if record.get("result", {}).get("status") == "ok":
                break
            consecutive_failures += 1
        if consecutive_failures >= int(
            config["budget"]["max_consecutive_failures"]
        ):
            raise SystemExit("Consecutive failure budget is exhausted")
        request = make_request(
            stage="agent_candidate",
            mode=args.mode,
            seed=args.seed,
            source_stations=list(config["source_stations"]),
            held_out_station=args.held_out,
            target_station=str(config["target_station"]),
            calibration_days=int(config["calibration_days"]),
            implementation_hash=config["_implementation_hash"],
            hypothesis=args.hypothesis,
            metadata={"data_contract": config["data_contract"]},
        )
        runner.run_one(request)
    elif args.command == "next":
        print(json.dumps(next_action(config, registry.records()), indent=2))
    elif args.command == "report":
        destination = state_dir / "reports" / "latest.md"
        write_report(registry.records(), config, destination)
        print(destination.read_text(encoding="utf-8"))
        print(f"\nreport: {destination}")
    elif args.command == "feedback-export":
        markdown, payload = export_feedback(
            registry.records(), config, state_dir / "feedback"
        )
        print(f"pseudonymized feedback markdown: {markdown}")
        print(f"pseudonymized feedback json: {payload}")
    elif args.command == "implementations":
        print(json.dumps(list_implementations(state_dir), indent=2))
    elif args.command == "rollback":
        if args.confirm != "RESTORE_EDITABLE":
            raise SystemExit("Rollback confirmation phrase is incorrect")
        restored = restore_implementation(
            ROOT / "editable", state_dir, args.implementation
        )
        print(f"restored editable implementation: {restored}")
    elif args.command == "final-test":
        if args.confirm != "RUN_FINAL_TARGET_ONCE":
            raise SystemExit("Final test confirmation phrase is incorrect")
        prior_final = [
            record
            for record in registry.records()
            if record.get("request", {}).get("final_test")
        ]
        if prior_final:
            raise SystemExit("A final target test is already recorded; refusing rerun")
        runner.run_one(final_request(config, args.mode, args.seed))


if __name__ == "__main__":
    main()
