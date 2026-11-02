"""Fraud metrics. Accuracy is never computed here: with a 3% fraud rate a model
that approves everything scores 97%. Everything below is imbalance-aware."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_curve


def recall_at_fpr(y_true, scores, max_fpr: float = 0.01) -> dict:
    """The headline metric (spec objective 5.2): how much fraud is caught while
    falsely flagging at most max_fpr of legitimate traffic. Returns the operating
    point, not just the number, because M4's threshold API needs the threshold."""
    y_true = np.asarray(y_true, int)
    scores = np.asarray(scores, float)
    if y_true.sum() == 0 or (y_true == 0).sum() == 0:
        return {"recall": 0.0, "fpr": 0.0, "threshold": float("inf"),
                "max_fpr": max_fpr}
    fpr, tpr, thr = roc_curve(y_true, scores)
    ok = fpr <= max_fpr
    i = int(np.argmax(np.where(ok, tpr, -1.0)))
    return {"recall": float(tpr[i]), "fpr": float(fpr[i]),
            "threshold": float(thr[i]), "max_fpr": max_fpr}


def pr_auc(y_true, scores) -> float:
    return float(average_precision_score(np.asarray(y_true, int),
                                         np.asarray(scores, float)))


def lift_over(baseline_recall: float, model_recall: float) -> float:
    """Relative improvement in recall at the same FPR - the number the spec's 30%
    objective is stated in."""
    return 0.0 if baseline_recall <= 0 else (model_recall - baseline_recall) / baseline_recall


def cost_curve(y_true, scores, amounts, review_cost: float = 3.0,
               friction_cost: float = 12.0, grid: int = 60) -> list[dict]:
    """Expected loss per threshold. Three costs, all real:
    - a missed fraud costs the transaction amount (chargeback),
    - a declined legitimate customer costs friction_cost (lost margin plus churn),
    - every flagged transaction costs review_cost of analyst time.
    The optimum is almost never 0.5, which is exactly the point."""
    y = np.asarray(y_true, int)
    s = np.asarray(scores, float)
    amt = np.asarray(amounts, float)
    out = []
    for t in np.linspace(s.min(), s.max(), grid):
        flag = s >= t
        missed = float(amt[(~flag) & (y == 1)].sum())
        friction = float(friction_cost * np.sum(flag & (y == 0)))
        review = float(review_cost * flag.sum())
        out.append({"threshold": float(t), "missed_fraud_value": missed,
                    "friction_cost": friction, "review_cost": review,
                    "total_cost": missed + friction + review,
                    "flag_rate": float(flag.mean())})
    return out


def best_threshold(curve: list[dict]) -> dict:
    return min(curve, key=lambda r: r["total_cost"])


def cohort_slices(y_true, scores, cohorts: dict, max_fpr: float = 0.01) -> dict:
    """recall@FPR inside each cohort (amount band, channel, new vs established).
    A model can hit its headline number and still be blind to one channel; this is
    how you find that before a fraud ring does."""
    out = {}
    for name, mask in cohorts.items():
        mask = np.asarray(mask, bool)
        if mask.sum() < 50 or np.asarray(y_true, int)[mask].sum() == 0:
            out[name] = {"n": int(mask.sum()), "recall": None,
                         "note": "too few rows or no fraud in cohort"}
            continue
        r = recall_at_fpr(np.asarray(y_true)[mask], np.asarray(scores)[mask], max_fpr)
        out[name] = {"n": int(mask.sum()), **r}
    return out


def amount_bands(amounts, edges=(0, 50, 250, 1000, np.inf)) -> dict:
    amt = np.asarray(amounts, float)
    return {f"amount_{int(lo)}_{'inf' if np.isinf(hi) else int(hi)}":
            (amt >= lo) & (amt < hi) for lo, hi in zip(edges[:-1], edges[1:])}
