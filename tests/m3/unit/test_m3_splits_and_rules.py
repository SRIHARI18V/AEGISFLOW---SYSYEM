"""Time splits, label delay and the rules baseline - the three places a fraud model
quietly reads the future or claims a lift it did not earn."""
import numpy as np
import pandas as pd
import pytest

from aegis_ml.evaluation import rules_baseline as rb
from aegis_ml.training.splits import (check_no_time_leak, label_delay_mask, time_split,
                                    week_index)

START = pd.Timestamp("2026-01-05", tz="UTC")


def frame(days=70):
    ts = pd.date_range(START, periods=days * 24, freq="h", tz="UTC")
    return pd.DataFrame({"ts": ts})


def test_week_index_counts_from_the_first_row():
    ts = frame(21)["ts"]
    w = week_index(ts)
    assert w.min() == 0 and w.max() == 2
    assert (w[ts < START + pd.Timedelta(days=7)] == 0).all()


def test_time_split_has_an_embargo_and_no_overlap():
    df = frame(70)
    df["split"] = time_split(df["ts"], train_weeks=6, test_weeks=2, embargo_hours=24)
    tr, te = df[df.split == "train"], df[df.split == "test"]
    assert len(tr) and len(te)
    assert check_no_time_leak(tr["ts"], te["ts"]) == []
    assert (te["ts"].min() - tr["ts"].max()) >= pd.Timedelta(hours=24)
    assert (df.split == "drop").sum() > 0 # the embargo rows


def test_leak_checker_fires_on_an_overlapping_split():
    df = frame(70)
    problems = check_no_time_leak(df["ts"], df["ts"])
    assert problems and "overlaps" in problems[0]
    assert check_no_time_leak(df["ts"], [])[0].startswith("train or test")


def test_label_delay_excludes_rows_whose_chargeback_has_not_landed():
    ts = pd.Series(pd.to_datetime(["2026-03-01", "2026-03-20", "2026-03-20"], utc=True))
    reported = pd.Series(pd.to_datetime([None, "2026-03-22", None], utc=True))
    known = label_delay_mask(ts, reported, as_of="2026-04-02", horizon_days=30)
    assert list(known) == [True, True, False]


def test_every_rule_has_a_positive_and_a_negative_case():
    quiet = pd.DataFrame({"amount": [100.0], "device_age_days": [400],
                          "txn_count_1h": [1], "amount_mean_30d": [120.0],
                          "distinct_countries_1h": [1], "device_account_degree": [1]})
    assert rb.score(quiet)[0] == 0.0
    assert rb.explain(quiet) == [[]]
    for rule in rb.RULES:
        hot = quiet.copy()
        hot.loc[0, {"R001": "amount", "R002": "device_age_days",
                    "R003": "txn_count_1h", "R004": "amount",
                    "R005": "distinct_countries_1h",
                    "R006": "device_account_degree"}[rule.id]] = {
            "R001": 60_000.0, "R002": 0, "R003": 9, "R004": 10_000.0,
            "R005": 3, "R006": 7}[rule.id]
        assert rule.id in rb.explain(hot)[0], f"{rule.id} never fires"


def test_rule_score_is_continuous_so_it_has_an_roc_curve():
    df = pd.DataFrame({"amount": [100.0, 60_000.0, 60_000.0],
                       "device_age_days": [400, 400, 0],
                       "txn_count_1h": [1, 1, 9],
                       "amount_mean_30d": [120.0, 120.0, 120.0],
                       "distinct_countries_1h": [1, 1, 3],
                       "device_account_degree": [1, 1, 1]})
    s = rb.score(df)
    assert s[0] < s[1] < s[2]
    assert rb.decisions(df).tolist() == [False, True, True]
