"""Objective 5.4: catch at least 80% of the injected laundering typologies.
The supervised model is trained on IEEE-CIS card fraud; smurfing and layering are
a different problem and live in the synthetic stream M1's generator produces with
ground truth. This module scores the generator's labelled typology episodes with
the champion model PLUS the graph features and reports detection per typology, so
the 80% claim is measured rather than asserted.
python -m aegis_ml.evaluation.typology_eval
"""
from __future__ import annotations

import json
import pathlib
import numpy as np
import pandas as pd

from .reports import write_report

# NOTE: training.matrix imports M2's feature definitions, which pull in Redis and
# the graph job. The import lives inside main() so episode_detection stays unit
# testable with nothing but pandas installed.
TYPOLOGIES = ["smurfing", "layering", "round_tripping"]
GROUND_TRUTH = pathlib.Path("data/processed/typology_episodes.parquet")


def episode_detection(episodes: pd.DataFrame, flagged_ids: set, min_share=0.30) -> dict:
    """An episode counts as detected when at least min_share of its member
    transactions were flagged. One lucky hit out of forty smurfing deposits is not
    a detection, and reporting it as one is how these numbers get inflated."""
    out = {}
    for typ in TYPOLOGIES:
        rows = episodes[episodes["typology"] == typ]
        if rows.empty:
            out[typ] = {"episodes": 0, "detected": 0, "rate": None}
            continue
        detected = 0
        for _, ep in rows.iterrows():
            members = set(map(str, ep["transaction_ids"]))
            if members and len(members & flagged_ids) / len(members) >= min_share:
                detected += 1
        out[typ] = {"episodes": int(len(rows)), "detected": detected,
                    "rate": detected / len(rows)}
    rates = [v["rate"] for v in out.values() if v["rate"] is not None]
    out["overall"] = {"rate": float(np.mean(rates)) if rates else 0.0}
    return out


def main() -> int:
    if not GROUND_TRUTH.exists():
        print(f"{GROUND_TRUTH} missing - run the M1 generator with typology injection "
              f"(`make generate`) before this evaluation")
        return 2
    import mlflow
    from ..training.matrix import build
    episodes = pd.read_parquet(GROUND_TRUTH)
    stream = pd.read_parquet("data/processed/synthetic_transactions.parquet")
    model = mlflow.sklearn.load_model("models:/aegisflow-fraud@champion")
    meta = json.loads(pathlib.Path("artifacts/fraud/metrics.json").read_text())
    X, _ = build(stream, mapping=meta["frequency_encoding"])
    probs = model.predict_proba(X)[:, 1]
    threshold = meta["results"][meta["champion"]]["threshold_at_1pct_fpr"]
    flagged = set(stream.loc[probs >= threshold, "transaction_id"].astype(str))
    result = episode_detection(episodes, flagged)
    overall = result["overall"]["rate"]
    gates = [("typology detection >= 80% (objective 5.4)", overall >= 0.80,
              f"{overall:.1%}")]
    write_report("reports/m3_typologies.html", "AegisFlow M3 - laundering typologies",
                 [{"heading": "Detection by typology",
                   "rows": [{"typology": k, **v} for k, v in result.items()
                            if k != "overall"],
                   "note": "An episode counts only when at least 30% of its member "
                           "transactions were flagged."}],
                 gates=gates, out_json="reports/m3_typologies.json")
    pathlib.Path("artifacts/fraud/typologies.json").write_text(
        json.dumps(result, indent=2, default=float))
    print(f"typology detection {overall:.1%} (gate 80%)")
    return 0 if overall >= 0.80 else 1


if __name__ == "__main__":
    raise SystemExit(main())
