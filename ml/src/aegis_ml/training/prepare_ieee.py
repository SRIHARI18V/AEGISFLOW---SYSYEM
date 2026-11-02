"""DVC stage: IEEE-CIS -> the canonical training frame (FR-1 data side).
IEEE-CIS ships TransactionDT as seconds from an unstated origin. The competition
convention is 2017-12-01; the absolute date does not matter, the ORDER does - every
split in this project is a time split, so a wrong origin would silently reorder the
weeks. The origin is written into the output metadata so it can never drift.
python -m aegis_ml.training.prepare_ieee --raw data/raw/ieee --out data/processed
"""
from __future__ import annotations

import argparse
import json
import pathlib
import pandas as pd

ORIGIN = "2017-12-01"
ID_COLS = ["TransactionID", "TransactionDT", "TransactionAmt", "ProductCD",
           "card1", "card4", "card6", "addr1", "P_emaildomain", "DeviceType",
           "DeviceInfo", "isFraud"]


def load(raw: pathlib.Path) -> pd.DataFrame:
    tx = pd.read_csv(raw / "train_transaction.csv")
    idf = raw / "train_identity.csv"
    if idf.exists():
        tx = tx.merge(pd.read_csv(idf), on="TransactionID", how="left")
    return tx


def canonicalise(df: pd.DataFrame, origin: str = ORIGIN) -> pd.DataFrame:
    out = pd.DataFrame({
        "transaction_id": df["TransactionID"].astype(str),
        "ts": pd.to_datetime(origin, utc=True)
            + pd.to_timedelta(df["TransactionDT"], unit="s"),
        "amount": df["TransactionAmt"].astype(float),
        "channel": df["ProductCD"].astype(str),
        "account_id": df["card1"].astype(str),
        "card_network": df.get("card4", pd.Series(index=df.index, dtype=object))
            .fillna("unknown").astype(str),
        "card_type": df.get("card6", pd.Series(index=df.index, dtype=object))
            .fillna("unknown").astype(str),
        "region": df.get("addr1", pd.Series(index=df.index)).astype("string")
            .fillna("unknown"),
        "email_domain": df.get("P_emaildomain",
                               pd.Series(index=df.index, dtype=object))
            .fillna("unknown").astype(str),
        "device_type": df.get("DeviceType", pd.Series(index=df.index, dtype=object))
            .fillna("unknown").astype(str),
        "device_id": df.get("DeviceInfo", pd.Series(index=df.index, dtype=object))
            .fillna("unknown").astype(str),
        "is_fraud": df["isFraud"].astype(int),
    })
    # chargebacks are not instant: give every labelled fraud a realistic reporting
    # delay so label_delay_mask has something honest to work with.
    # We ensure 2-digit slice works reliably for any id format
    clean_id = out["transaction_id"].str.replace(r"\D", "", regex=True)
    clean_id = clean_id.where(clean_id.str.len() >= 2, "42")
    delay = pd.to_timedelta(
        (clean_id.str.slice(-2).astype(int) % 45) + 3, unit="D")
    out["label_reported_at"] = out["ts"] + delay.where(out["is_fraud"] == 1)
    return out.sort_values("ts").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/ieee")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--origin", default=ORIGIN)
    a = ap.parse_args()
    raw, out = pathlib.Path(a.raw), pathlib.Path(a.out)
    if not (raw / "train_transaction.csv").exists():
        raise SystemExit(f"{raw}/train_transaction.csv missing - download the "
                         f"IEEE-CIS competition data (Kaggle credentials required)")
    df = canonicalise(load(raw), a.origin)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "ieee_transactions.parquet", index=False)
    (out / "ieee_meta.json").write_text(json.dumps({
        "origin": a.origin, "rows": int(len(df)),
        "fraud_rate": float(df["is_fraud"].mean()),
        "first_ts": str(df["ts"].min()), "last_ts": str(df["ts"].max()),
        "weeks": int((df["ts"].max() - df["ts"].min()).days // 7) + 1}, indent=2))
    print(f"{len(df)} rows, fraud rate {df['is_fraud'].mean():.3%}, "
          f"{df['ts'].min().date()} to {df['ts'].max().date()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
