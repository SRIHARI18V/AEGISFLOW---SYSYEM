"""Time-based splits. There is exactly one correct way to split fraud data.
Fraud is adversarial and seasonal: rings appear, get blocked, and are replaced.
A random split lets the model see next week's ring while training, so it scores
beautifully offline and collapses in production. Train on weeks 1-6, test on 7-8
(spec section 14), and never shuffle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRAIN_WEEKS = 6
TEST_WEEKS = 2


def week_index(ts: pd.Series, origin=None) -> pd.Series:
    """0-based week number from the first transaction in the dataset."""
    ts = pd.to_datetime(ts, utc=True)
    origin = pd.to_datetime(origin, utc=True) if origin is not None else ts.min()
    return ((ts - origin).dt.total_seconds() // (7 * 86400)).astype(int)


def time_split(ts: pd.Series, train_weeks: int = TRAIN_WEEKS,
               test_weeks: int = TEST_WEEKS, embargo_hours: int = 24) -> pd.Series:
    """'train' / 'test' / 'drop'. The embargo removes the hours straddling the
    boundary: a transaction scored just before the cut shares velocity windows with
    transactions just after it, which is leakage through the feature store rather
    than through the label."""
    w = week_index(ts)
    ts = pd.to_datetime(ts, utc=True)
    cut = ts.min() + pd.Timedelta(weeks=train_weeks)
    out = pd.Series("drop", index=ts.index, dtype=object)
    out[(w < train_weeks) & (ts < cut - pd.Timedelta(hours=embargo_hours))] = "train"
    out[(w >= train_weeks) & (w < train_weeks + test_weeks)] = "test"
    return out


def check_no_time_leak(train_ts, test_ts) -> list[str]:
    """Empty list means clean: every training timestamp is strictly before every
    test timestamp."""
    tr, te = pd.to_datetime(train_ts, utc=True), pd.to_datetime(test_ts, utc=True)
    problems = []
    if len(tr) == 0 or len(te) == 0:
        problems.append("train or test split is empty")
        return problems
    if tr.max() >= te.min():
        problems.append(
            f"train ends {tr.max()} but test starts {te.min()} - the split overlaps")
        return problems
    return problems


def label_delay_mask(ts, reported_at, as_of, horizon_days: int = 30) -> np.ndarray:
    """Which rows had a KNOWN label at `as_of`. Chargebacks arrive weeks late, so a
    model 'evaluated' on labels that did not exist yet is reading the future. Rows
    whose label had not landed are excluded from training, not silently marked legit."""
    ts = pd.to_datetime(ts, utc=True)
    reported = pd.to_datetime(reported_at, utc=True)
    as_of = pd.Timestamp(as_of)
    as_of = as_of.tz_localize("UTC") if as_of.tzinfo is None else as_of.tz_convert("UTC")
    matured = ts <= as_of - pd.Timedelta(days=horizon_days)
    known = reported.notna() & (reported <= as_of)
    return (matured | known).to_numpy()
