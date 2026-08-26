from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def build_report(records: list[dict[str, Any]], config: dict) -> str:
    legacy_count = sum(
        1
        for record in records
        if record.get("result", {}).get("status") == "ok"
        and not record.get("request", {}).get("final_test", False)
        and not record.get("request", {}).get("implementation_hash")
    )
    successful = [
        record
        for record in records
        if record.get("result", {}).get("status") == "ok"
        and not record.get("request", {}).get("final_test", False)
        and record.get("request", {}).get("implementation_hash")
    ]
    if not successful:
        return "# PV Registration Report\n\nNo successful experiments.\n"

    available_stages = {
        record["request"]["stage"] for record in successful
    }
    selected_stage = (
        "full_loso"
        if "full_loso" in available_stages
        else "quick_ablation"
        if "quick_ablation" in available_stages
        else "identity_audit"
    )
    research_records = [
        record
        for record in successful
        if record["request"]["stage"] == selected_stage
    ]

    baseline_by_pair: dict[tuple, float] = {}
    for record in research_records:
        request = record["request"]
        if request["mode"] == "baseline":
            baseline_by_pair[_pair(request)] = float(record["result"]["score"])

    mode_scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    mode_gains: dict[tuple[str, str], list[float]] = defaultdict(list)
    station_gains: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in research_records:
        request = record["request"]
        mode = str(request["mode"])
        implementation = str(request["implementation_hash"])
        key = (mode, implementation)
        score = float(record["result"]["score"])
        mode_scores[key].append(score)
        baseline = baseline_by_pair.get(_pair(request))
        if baseline is not None and mode != "baseline":
            gain = score - baseline
            mode_gains[key].append(gain)
            station_gains[key][request["held_out_station"]].append(gain)

    lines = [
        "# PV Registration Report",
        "",
        "Final target-test records are intentionally excluded.",
        f"Primary comparison stage: `{selected_stage}`.",
        *(
            [
                f"Ignored {legacy_count} successful legacy record(s) without "
                "an implementation hash."
            ]
            if legacy_count
            else []
        ),
        "",
        "| mode | implementation | runs | mean score | mean gain | worst station gain | positive station ratio | decision |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    acceptance = config["acceptance"]
    for key in sorted(mode_scores):
        mode, implementation = key
        short_implementation = implementation[:12]
        scores = mode_scores[key]
        gains = mode_gains.get(key, [])
        if not gains:
            lines.append(
                f"| {mode} | {short_implementation} | {len(scores)} | "
                f"{statistics.mean(scores):.6f} "
                "| n/a | n/a | n/a | reference |"
            )
            continue
        per_station = {
            station: statistics.mean(values)
            for station, values in station_gains[key].items()
        }
        mean_gain = statistics.mean(gains)
        worst = min(per_station.values())
        positive_ratio = sum(x > 0 for x in per_station.values()) / len(per_station)
        accepted = (
            mean_gain >= float(acceptance["min_mean_ood_gain"])
            and worst >= -float(acceptance["max_worst_station_drop"])
            and positive_ratio >= float(acceptance["min_positive_station_ratio"])
        )
        lines.append(
            f"| {mode} | {short_implementation} | {len(scores)} | "
            f"{statistics.mean(scores):.6f} "
            f"| {mean_gain:+.6f} | {worst:+.6f} | {positive_ratio:.2%} "
            f"| {'accept' if accepted else 'reject'} |"
        )

    lines += ["", "## Per-station paired gains", ""]
    for key in sorted(station_gains):
        mode, implementation = key
        lines.append(f"### {mode} @ {implementation[:12]}")
        lines.append("")
        for station, values in sorted(station_gains[key].items()):
            lines.append(f"- {station}: {statistics.mean(values):+.6f}")
        lines.append("")

    audit = {
        record["request"]["mode"]: float(record["result"]["score"])
        for record in successful
        if record["request"]["stage"] == "identity_audit"
    }
    if "baseline" in audit and "identity" in audit:
        gap = audit["identity"] - audit["baseline"]
        tolerance = float(config["identity_score_tolerance"])
        lines += [
            "## Identity parity",
            "",
            f"- identity - baseline: {gap:+.6f}",
            f"- tolerance: {tolerance:.6f}",
            f"- result: {'pass' if abs(gap) <= tolerance else 'fail'}",
            "",
        ]
    return "\n".join(lines)


def write_report(
    records: list[dict[str, Any]], config: dict, destination: Path
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(build_report(records, config), encoding="utf-8")
    return destination


def _pair(request: dict[str, Any]) -> tuple:
    return (
        request["stage"],
        request["held_out_station"],
        int(request["seed"]),
    )
