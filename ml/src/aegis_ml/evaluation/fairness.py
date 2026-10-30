"""Disparate-impact screens (spec section 14, 'fairness screens').
This is not a legal compliance tool and the guide says so out loud. It answers one
narrow question: does the model decline some cohorts at a much higher rate than
others at the SAME operating threshold, and is that gap explained by the fraud rate
in the cohort or not? An unexplained gap is a finding for the model card, not an
automatic failure.
"""
from __future__ import annotations

import numpy as np

FOUR_FIFTHS = 0.80


def group_rates(y_true, flags, groups: dict) -> dict:
    out = {}
    y = np.asarray(y_true, int)
    f = np.asarray(flags, bool)
    for name, mask in groups.items():
        m = np.asarray(mask, bool)
        if m.sum() < 50:
            out[name] = {"n": int(m.sum()), "note": "cohort too small to screen"}
            continue
        legit = m & (y == 0)
        out[name] = {"n": int(m.sum()),
                     "flag_rate": float(f[m].mean()),
                     "false_positive_rate": float(f[legit].mean()) if legit.any() else None,
                     "fraud_rate": float(y[m].mean())}
    return out


def disparate_impact(rates: dict, key: str = "flag_rate") -> dict:
    """Ratio of the lowest to the highest cohort rate. Below 0.8 is the classic
    four-fifths flag used in US adverse-impact analysis."""
    vals = {k: v[key] for k, v in rates.items() if v.get(key) is not None}
    if len(vals) < 2:
        return {"ratio": None, "note": "need at least two screenable cohorts"}
    lo_k, lo = min(vals.items(), key=lambda kv: kv[1])
    hi_k, hi = max(vals.items(), key=lambda kv: kv[1])
    ratio = (lo / hi) if hi > 0 else None
    return {"ratio": ratio, "lowest": lo_k, "highest": hi_k,
            "flag": ratio is not None and ratio < FOUR_FIFTHS,
            "note": "a gap tracking the cohort fraud rate is expected; "
                    "an unexplained gap goes in the model card"}
