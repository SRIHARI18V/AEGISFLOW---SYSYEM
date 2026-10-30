"""Promotion gates, with no MLflow import so they are testable on a laptop.
Every bound traces to a line in the spec."""
from __future__ import annotations

GATES: dict[str, tuple[bool, float, str]] = {
    "recall_at_1pct_fpr": (False, 0.50, "section 19 - absolute floor before anything ships"),
    "lift_over_rules": (False, 0.30, "objective 5.2 - beat the rules baseline by >= 30%"),
    "pr_auc": (False, 0.60, "section 14 - headline ranking quality"),
    "ece": (True, 0.05, "section 14 - calibrated within 5 points"),
    "typology_detection": (False, 0.80, "objective 5.4 - >= 80% of injected typologies"),
}


def check_gate(metric: str, value: float) -> tuple[bool, str]:
    if metric not in GATES:
        return True, f"{metric}={value:.4f} (no gate defined)"
    lower_better, bound, ref = GATES[metric]
    ok = value <= bound if lower_better else value >= bound
    rel = "<=" if lower_better else ">="
    return ok, f"{metric}={value:.4f} {rel} {bound} [{ref}] -> {'pass' if ok else 'FAIL'}"


def is_better(metric: str, candidate: float, incumbent: float) -> bool:
    """Strictly better, in the metric's own direction. A tie is churn, not a win."""
    lower_better = GATES.get(metric, (False, 0.0, ""))[0]
    return candidate < incumbent if lower_better else candidate > incumbent


def challenger_wins(challenger: dict, champion: dict,
                    primary: str = "recall_at_1pct_fpr",
                    min_relative_lift: float = 0.02) -> tuple[bool, str]:
    """Champion/challenger promotion rule (FR-7). A challenger has to clear every
    gate AND beat the champion on the primary metric by a margin, because a 0.2%
    'improvement' inside the noise is not worth a production swap."""
    failed = [check_gate(m, v)[1] for m, v in challenger.items()
              if m in GATES and not check_gate(m, v)[0]]
    if failed:
        return False, "challenger failed gates: " + "; ".join(failed)
    c, p = challenger.get(primary), champion.get(primary)
    if c is None or p is None:
        return False, f"{primary} missing from one of the models"
    if not is_better(primary, c, p):
        return False, f"{primary}: challenger {c:.4f} does not beat champion {p:.4f}"
    lift = abs(c - p) / max(abs(p), 1e-9)
    if lift < min_relative_lift:
        return False, (f"{primary}: {lift:.2%} relative lift is inside the noise "
                       f"band ({min_relative_lift:.0%})")
    return True, f"{primary}: {p:.4f} -> {c:.4f} ({lift:+.2%}) - promote"
