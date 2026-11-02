"""Assemble the training matrix from M2's feature definitions.
The rule this file exists to enforce: M3 must not invent features. Every column
comes from ml/src/aegis_ml/features/definitions.py - the same declarations that
compile to the Redis update path - so whatever the model learns, the scoring
service in M4 can actually compute in under 100 ms. If a feature is missing here,
the fix is to declare it in definitions.py, not to add a pandas expression.
"""
from __future__ import annotations

import pathlib
import numpy as np
import pandas as pd

from ..features.definitions import FEATURES # M2 - the single source
from ..features.offline import build_point_in_time # M2 - batch compiler
from ..features.graph_features import load_graph_features # M2 - NetworkX job output

PROCESSED = pathlib.Path("data/processed")
CATEGORICAL = ["channel", "card_network", "card_type", "device_type",
               "email_domain", "region"]


def load_transactions(path=PROCESSED / "ieee_transactions.parquet") -> pd.DataFrame:
    if not pathlib.Path(path).exists():
        raise FileNotFoundError(
            f"{path} missing - run `dvc repro prepare_ieee` (M3) and the synthetic "
            f"generator (M1) first")
    return pd.read_parquet(path)


def feature_names() -> list[str]:
    """Declared features + graph features + encoded categoricals, in a FIXED order.
    The order is persisted with the model: a column shuffle between training and
    serving is silent and catastrophic."""
    all_names = ([f.name for f in FEATURES]
        + ["graph_component_size", "graph_dist_to_fraud", "graph_risk_rank",
           "graph_device_degree"]
        + [f"cat__{c}" for c in CATEGORICAL]
        + ["amount", "log_amount", "hour_of_day", "day_of_week"])
    seen = set()
    return [c for c in all_names if not (c in seen or seen.add(c))]


def encode_categoricals(df: pd.DataFrame, mapping: dict | None = None):
    """Frequency encoding fitted on TRAIN ONLY. Target encoding would leak the label
    through the encoder, which is the most common silent leak in fraud notebooks."""
    mapping = mapping if mapping is not None else {
        c: df[c].astype(str).value_counts(normalize=True).to_dict() for c in CATEGORICAL}
    out = pd.DataFrame(index=df.index)
    for c in CATEGORICAL:
        out[f"cat__{c}"] = df[c].astype(str).map(mapping[c]).fillna(0.0)
    return out, mapping


def build(df: pd.DataFrame, as_of_column: str = "ts", mapping: dict | None = None):
    """Returns (X DataFrame in feature_names() order, mapping). Point-in-time
    correctness is M2's job; this function only refuses to proceed without it."""
    pit = build_point_in_time(df, FEATURES, as_of_column=as_of_column)
    if len(pit) != len(df):
        raise AssertionError("point-in-time builder returned a different row count - "
                             "a join fanned out; fix it before training")
    graph = load_graph_features(df["account_id"], as_of=df[as_of_column])
    cats, mapping = encode_categoricals(df, mapping)
    base = pd.DataFrame({
        "amount": df["amount"].astype(float),
        "log_amount": np.log1p(df["amount"].astype(float)),
        "hour_of_day": pd.to_datetime(df[as_of_column], utc=True).dt.hour,
        "day_of_week": pd.to_datetime(df[as_of_column], utc=True).dt.dayofweek,
    }, index=df.index)
    X = pd.concat([pit, graph, cats, base], axis=1)
    X = X.loc[:, ~X.columns.duplicated()]
    missing = [c for c in feature_names() if c not in X.columns]
    if missing:
        raise KeyError(f"declared features missing from the built matrix: {missing}")
    return X[feature_names()], mapping
