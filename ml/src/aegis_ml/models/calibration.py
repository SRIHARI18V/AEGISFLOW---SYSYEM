"""Isotonic calibration and the reliability check (spec section 14).
A raw XGBoost margin is a ranking, not a probability. Cost-based decisioning
multiplies the score by a transaction amount, so a score of 0.9 has to mean roughly
a 90% chance of fraud or every cost number downstream is nonsense. Isotonic
regression is fitted on a held-out slice, never on the training rows.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


class Calibrator:
    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.fitted = False

    def fit(self, scores, y_true) -> "Calibrator":
        self.iso.fit(np.asarray(scores, float), np.asarray(y_true, float))
        self.fitted = True
        return self

    def transform(self, scores) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("calibrator not fitted")
        return np.clip(self.iso.predict(np.asarray(scores, float)), 0.0, 1.0)


def reliability(y_true, probs, bins: int = 10) -> list[dict]:
    """Per-bin predicted vs observed fraud rate - the table behind the reliability
    curve in the HTML report."""
    y = np.asarray(y_true, int)
    p = np.asarray(probs, float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if not m.any():
            continue
        out.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": int(m.sum()),
                    "predicted": float(p[m].mean()), "observed": float(y[m].mean())})
    return out


def expected_calibration_error(y_true, probs, bins: int = 10) -> float:
    """Weighted mean gap between predicted and observed rates. Spec section 14 asks
    for calibration; this is the single number that summarises it."""
    rows = reliability(y_true, probs, bins)
    n = sum(r["n"] for r in rows) or 1
    return float(sum(r["n"] * abs(r["predicted"] - r["observed"]) for r in rows) / n)
