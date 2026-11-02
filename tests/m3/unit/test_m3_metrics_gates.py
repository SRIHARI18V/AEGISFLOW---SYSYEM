"""Metrics, calibration, fairness and the champion/challenger rule."""
import numpy as np
import pytest

from aegis_ml.evaluation.fairness import disparate_impact, group_rates
from aegis_ml.evaluation.gates import challenger_wins, check_gate, is_better
from aegis_ml.evaluation.metrics import (amount_bands, best_threshold, cohort_slices,
                                        cost_curve, lift_over, pr_auc, recall_at_fpr)
from aegis_ml.models.calibration import (Calibrator, expected_calibration_error,
                                        reliability)

RNG = np.random.default_rng(7)


def sample(n=4000, rate=0.03, sep=2.5):
    y = (RNG.random(n) < rate).astype(int)
    s = RNG.normal(0, 1, n) + sep * y
    amt = np.exp(RNG.normal(4, 1, n))
    return y, s, amt


def test_recall_at_fpr_respects_the_budget():
    y, s, _ = sample()
    r = recall_at_fpr(y, s, 0.01)
    assert r["fpr"] <= 0.01 + 1e-9
    assert 0.0 < r["recall"] <= 1.0
    loose = recall_at_fpr(y, s, 0.10)
    assert loose["recall"] >= r["recall"] # a bigger budget never catches less


def test_recall_at_fpr_handles_a_degenerate_label_vector():
    r = recall_at_fpr([0, 0, 0], [0.1, 0.2, 0.3], 0.01)
    assert r["recall"] == 0.0 and r["threshold"] == float("inf")


def test_pr_auc_and_lift():
    y, s, _ = sample()
    assert pr_auc(y, s) > pr_auc(y, RNG.normal(0, 1, len(y)))
    assert lift_over(0.40, 0.52) == pytest.approx(0.30)
    assert lift_over(0.0, 0.5) == 0.0


def test_cost_curve_optimum_is_not_the_midpoint():
    y, s, amt = sample()
    curve = cost_curve(y, s, amt, review_cost=3.0, friction_cost=12.0)
    best = best_threshold(curve)
    assert best["total_cost"] == min(r["total_cost"] for r in curve)
    assert 0.0 <= best["flag_rate"] <= 1.0
    # a cheaper review makes it worth flagging more
    cheap = best_threshold(cost_curve(y, s, amt, review_cost=0.1, friction_cost=1.0))
    assert cheap["flag_rate"] >= best["flag_rate"]


def test_cohort_slices_flag_thin_cohorts_instead_of_inventing_numbers():
    y, s, amt = sample()
    out = cohort_slices(y, s, amount_bands(amt), 0.01)
    assert set(out) == {"amount_0_50", "amount_50_250", "amount_250_1000",
                        "amount_1000_inf"}
    tiny = cohort_slices(y, s, {"tiny": np.r_[np.ones(10, bool),
                                              np.zeros(len(y) - 10, bool)]})
    assert tiny["tiny"]["recall"] is None and "too few" in tiny["tiny"]["note"]


def test_isotonic_calibration_fixes_a_squashed_score():
    y, s, _ = sample(n=8000, rate=0.10)
    probs_raw = 1 / (1 + np.exp(-(s * 0.25))) # deliberately under-confident
    cal = Calibrator().fit(probs_raw[:4000], y[:4000])
    probs_cal = cal.transform(probs_raw[4000:])
    assert expected_calibration_error(y[4000:], probs_cal) < \
        expected_calibration_error(y[4000:], probs_raw[4000:])
    rows = reliability(y[4000:], probs_cal)
    assert rows and all(0.0 <= r["observed"] <= 1.0 for r in rows)
    with pytest.raises(RuntimeError):
        Calibrator().transform([0.5])


def test_disparate_impact_uses_the_four_fifths_rule():
    y = np.r_[np.zeros(500, int), np.zeros(500, int)]
    flags = np.r_[np.ones(100, bool), np.zeros(400, bool),
                  np.ones(400, bool), np.zeros(100, bool)]
    groups = {"a": np.r_[np.ones(500, bool), np.zeros(500, bool)],
              "b": np.r_[np.zeros(500, bool), np.ones(500, bool)]}
    rates = group_rates(y, flags, groups)
    di = disparate_impact(rates)
    assert di["ratio"] == pytest.approx(0.25) and di["flag"] is True
    assert di["lowest"] == "a" and di["highest"] == "b"
    assert disparate_impact({"a": {"flag_rate": 0.1}})["ratio"] is None


def test_gates_and_the_noise_band():
    assert check_gate("ece", 0.04)[0] and not check_gate("ece", 0.06)[0]
    assert check_gate("lift_over_rules", 0.30)[0]
    assert not is_better("ece", 0.05, 0.05)
    champ = {"recall_at_1pct_fpr": 0.52}
    ok, why = challenger_wins({"recall_at_1pct_fpr": 0.60, "lift_over_rules": 0.35,
                               "pr_auc": 0.71, "ece": 0.03}, champ)
    assert ok and "promote" in why
    tiny, why2 = challenger_wins({"recall_at_1pct_fpr": 0.525, "lift_over_rules": 0.35,
                                  "pr_auc": 0.71, "ece": 0.03}, champ)
    assert not tiny and "noise band" in why2
    bad, why3 = challenger_wins({"recall_at_1pct_fpr": 0.60, "lift_over_rules": 0.35,
                                 "pr_auc": 0.71, "ece": 0.20}, champ)
    assert not bad and "ece" in why3


def test_typology_episode_needs_more_than_one_lucky_hit():
    import pandas as pd
    from aegis_ml.evaluation.typology_eval import episode_detection
    eps = pd.DataFrame([{"typology": "smurfing", "transaction_ids": list("abcd")},
                        {"typology": "layering", "transaction_ids": list("ef")},
                        {"typology": "round_tripping", "transaction_ids": list("gh")}])
    two_of_four = episode_detection(eps, set("ab"))
    assert two_of_four["smurfing"] == {"episodes": 1, "detected": 1, "rate": 1.0}
    one_of_four = episode_detection(eps, set("a"))
    assert one_of_four["smurfing"]["detected"] == 0 # 25% < the 30% share rule
    assert one_of_four["overall"]["rate"] == 0.0
