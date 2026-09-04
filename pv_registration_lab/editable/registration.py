"""Low-degree-of-freedom registration candidates.

This file is intentionally editable by the in-environment research agent.
Keep public interfaces stable so protected evaluation remains comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np


MINUTES_PER_DAY = 1440.0
DEFAULT_KNOTS = np.array([0.0, 0.25, 0.50, 0.75, 1.0])


@dataclass(frozen=True)
class MonotoneWarp:
    canonical_knots: np.ndarray
    physical_knots: np.ndarray
    name: str = "custom"

    def __post_init__(self):
        canonical = np.asarray(self.canonical_knots, dtype=np.float64)
        physical = np.asarray(self.physical_knots, dtype=np.float64)
        if canonical.ndim != 1 or canonical.shape != physical.shape:
            raise ValueError("Warp knots must be equal-length 1D arrays")
        if len(canonical) < 2:
            raise ValueError("At least two warp knots are required")
        if canonical[0] != 0.0 or canonical[-1] != 1.0:
            raise ValueError("Canonical endpoints must be 0 and 1")
        if physical[0] != 0.0 or physical[-1] != 1.0:
            raise ValueError("Physical endpoints must be 0 and 1")
        if np.any(np.diff(canonical) <= 0) or np.any(np.diff(physical) <= 0):
            raise ValueError("Warp knots must be strictly increasing")
        object.__setattr__(self, "canonical_knots", canonical)
        object.__setattr__(self, "physical_knots", physical)

    @classmethod
    def identity(cls) -> "MonotoneWarp":
        return cls(DEFAULT_KNOTS.copy(), DEFAULT_KNOTS.copy(), "identity")

    def canonical_to_physical(self, canonical_fraction):
        return np.interp(
            canonical_fraction,
            self.canonical_knots,
            self.physical_knots,
        )

    def physical_to_canonical(self, physical_fraction):
        return np.interp(
            physical_fraction,
            self.physical_knots,
            self.canonical_knots,
        )

    def local_slopes(self) -> np.ndarray:
        return np.diff(self.physical_knots) / np.diff(self.canonical_knots)

    def max_shift_minutes(self) -> float:
        return float(
            np.max(np.abs(self.physical_knots - self.canonical_knots))
            * MINUTES_PER_DAY
        )

    def apply_curve(self, curve: np.ndarray) -> np.ndarray:
        curve = np.asarray(curve, dtype=np.float64)
        grid = np.arange(len(curve), dtype=np.float64) / len(curve)
        physical = self.canonical_to_physical(grid)
        return np.interp(
            physical,
            np.r_[grid, 1.0],
            np.r_[curve, curve[0]],
        )

    def inverse_curve(self, registered: np.ndarray) -> np.ndarray:
        registered = np.asarray(registered, dtype=np.float64)
        grid = np.arange(len(registered), dtype=np.float64) / len(registered)
        canonical = self.physical_to_canonical(grid)
        return np.interp(
            canonical,
            np.r_[grid, 1.0],
            np.r_[registered, registered[0]],
        )


def normalize_shape(
    curve: np.ndarray, daylight_peak_ratio: float = 0.02
) -> np.ndarray:
    curve = np.clip(np.asarray(curve, dtype=np.float64), 0.0, None)
    peak = float(curve.max())
    if peak <= 1e-8:
        raise ValueError("Invalid curve peak")
    # A relative threshold is required for amplitude-invariant shape fitting.
    # An absolute capacity-ratio threshold selects different points for two
    # otherwise identical curves with different stable amplitudes.
    daylight = curve > peak * daylight_peak_ratio
    if daylight.sum() < 8:
        raise ValueError("Too few daylight points")
    scale = float(np.quantile(curve[daylight], 0.95))
    if scale <= 1e-8:
        raise ValueError("Invalid shape scale")
    return curve / scale


def shape_rmse(curve: np.ndarray, template: np.ndarray) -> float:
    curve = normalize_shape(curve)
    template = normalize_shape(template)
    daylight = (curve > 0.02) | (template > 0.02)
    return float(np.sqrt(np.mean((curve[daylight] - template[daylight]) ** 2)))


def fit_translation(
    curve: np.ndarray,
    template: np.ndarray,
    *,
    max_shift_minutes: int = 30,
    step_minutes: int = 15,
) -> MonotoneWarp:
    """Fit one station phase shift; amplitude is ignored during fitting."""
    shifts = range(-max_shift_minutes, max_shift_minutes + 1, step_minutes)
    best = MonotoneWarp.identity()
    best_loss = shape_rmse(curve, template)
    for shift in shifts:
        delta = shift / MINUTES_PER_DAY
        physical = DEFAULT_KNOTS.copy()
        physical[1:-1] += delta
        if np.any(np.diff(physical) <= 0):
            continue
        warp = MonotoneWarp(DEFAULT_KNOTS.copy(), physical, "translation")
        loss = shape_rmse(warp.apply_curve(curve), template)
        if loss < best_loss:
            best, best_loss = warp, loss
    return best


def fit_three_point(
    curve: np.ndarray,
    template: np.ndarray,
    *,
    max_shift_minutes: int = 30,
    step_minutes: int = 15,
    min_slope: float = 0.85,
    max_slope: float = 1.18,
    identity_penalty: float = 0.02,
    min_relative_improvement: float = 0.01,
) -> MonotoneWarp:
    """Fit morning/noon/evening anchors with strict monotonic constraints."""
    baseline = shape_rmse(curve, template)
    candidates = range(-max_shift_minutes, max_shift_minutes + 1, step_minutes)
    best = MonotoneWarp.identity()
    best_objective = baseline
    for shifts in product(candidates, repeat=3):
        physical = DEFAULT_KNOTS.copy()
        physical[1:-1] += np.asarray(shifts) / MINUTES_PER_DAY
        if np.any(np.diff(physical) <= 0):
            continue
        warp = MonotoneWarp(DEFAULT_KNOTS.copy(), physical, "three_point")
        slopes = warp.local_slopes()
        if slopes.min() < min_slope or slopes.max() > max_slope:
            continue
        loss = shape_rmse(warp.apply_curve(curve), template)
        penalty = identity_penalty * np.mean(
            (warp.physical_knots - warp.canonical_knots) ** 2
        )
        objective = loss + penalty
        if objective < best_objective:
            best, best_objective = warp, objective
    after = shape_rmse(best.apply_curve(curve), template)
    improvement = (baseline - after) / baseline if baseline > 0 else 0.0
    if improvement < min_relative_improvement:
        return MonotoneWarp.identity()
    return best


def season_for_month(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    if month in (9, 10, 11):
        return "autumn"
    raise ValueError(f"Invalid month: {month}")


@dataclass(frozen=True)
class SeasonalWarps:
    by_season: dict[str, MonotoneWarp]

    def for_month(self, month: int) -> MonotoneWarp:
        season = season_for_month(month)
        return self.by_season.get(season, MonotoneWarp.identity())
