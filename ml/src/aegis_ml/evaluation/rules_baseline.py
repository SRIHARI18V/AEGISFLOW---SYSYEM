"""The rule engine the platform must beat (spec problem statement: legacy rules
produce 90%+ false positives). Implemented honestly, with the thresholds a real
bank would set, so the 30% lift claim in objective 5.2 is a measurement.
M4's services/scoring/rules.py reuses these pure functions as its deterministic
layer - the rules do not disappear when the model arrives, they become the floor
that can force a decline regardless of score.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Rule:
    id: str
    description: str
    predicate: Callable[[pd.DataFrame], np.ndarray]
    weight: float = 1.0


RULES: list[Rule] = [
    Rule("R001", "amount above 50,000",
         lambda d: d["amount"].to_numpy() > 50_000, 1.0),
    Rule("R002", "new device for this account",
         lambda d: d.get("device_age_days", pd.Series(99, index=d.index)
                         ).to_numpy() < 1, 1.0),
    Rule("R003", "more than 5 transactions in the last hour",
         lambda d: d.get("txn_count_1h", pd.Series(0, index=d.index)).to_numpy() > 5, 1.0),
    Rule("R004", "amount more than 5x the account's 30-day average",
         lambda d: d["amount"].to_numpy() >
         5 * np.maximum(d.get("amount_mean_30d",
                              pd.Series(1.0, index=d.index)).to_numpy(), 1.0), 1.0),
    Rule("R005", "impossible travel: two countries within an hour",
         lambda d: d.get("distinct_countries_1h",
                         pd.Series(1, index=d.index)).to_numpy() > 1, 2.0),
    Rule("R006", "device shared by more than 3 accounts",
         lambda d: d.get("device_account_degree",
                         pd.Series(1, index=d.index)).to_numpy() > 3, 1.0),
]


def hits(df: pd.DataFrame) -> dict:
    """Rule id -> boolean array. Pure, so each rule gets its own unit test with a
    positive and a negative fixture, which is what makes the layer auditable."""
    return {r.id: np.asarray(r.predicate(df), bool) for r in RULES}


def score(df: pd.DataFrame) -> np.ndarray:
    """Weighted count of rule hits, used as a continuous score so the baseline has a
    ROC curve and can be compared at the same FPR as the model."""
    fired = hits(df)
    total = np.zeros(len(df), dtype=float)
    for r in RULES:
        total += r.weight * fired[r.id]
    return total


def decisions(df: pd.DataFrame, min_weight: float = 1.0) -> np.ndarray:
    return score(df) >= min_weight


def explain(df: pd.DataFrame) -> list[list[str]]:
    fired = hits(df)
    return [[r.id for r in RULES if fired[r.id][i]] for i in range(len(df))]
