"""Generate synthetic IEEE canonical data and typology stream for M3."""
from __future__ import annotations

import json
import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data/processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

START_DATE = "2017-12-01"


def generate_ieee_data(n=60000, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start_ts = pd.Timestamp(START_DATE, tz="UTC")
    
    # 57 days (8 weeks + 1 day)
    total_seconds = 57 * 86400
    offsets = np.sort(rng.uniform(0, total_seconds, size=n))
    ts = start_ts + pd.to_timedelta(offsets, unit="s")
    
    # Accounts
    num_accounts = 3000
    account_ids = [f"acct_{i}" for i in range(num_accounts)]
    acc_probs = 1.0 / np.arange(1, num_accounts + 1) ** 0.8
    acc_probs /= acc_probs.sum()
    chosen_accounts = rng.choice(account_ids, size=n, p=acc_probs)
    
    # Devices & IPs
    num_devices = 2000
    device_ids = [f"dev_{i}" for i in range(num_devices)]
    chosen_devices = rng.choice(device_ids, size=n)
    chosen_devices = [d if rng.random() > 0.10 else "unknown" for d in chosen_devices]
    
    # Categoricals
    channels = ["W", "C", "R", "H", "S"]
    card_networks = ["visa", "mastercard", "discover", "amex"]
    card_types = ["debit", "credit"]
    regions = [str(r) for r in [120, 204, 315, 440, 512, 180]]
    email_domains = ["gmail.com", "yahoo.com", "hotmail.com", "anonymous.info", "mail.com"]
    device_types = ["desktop", "mobile"]
    
    chosen_channels = rng.choice(channels, size=n, p=[0.70, 0.12, 0.08, 0.06, 0.04])
    chosen_networks = rng.choice(card_networks, size=n, p=[0.60, 0.30, 0.06, 0.04])
    chosen_card_types = rng.choice(card_types, size=n, p=[0.72, 0.28])
    chosen_regions = rng.choice(regions, size=n)
    chosen_domains = rng.choice(email_domains, size=n, p=[0.45, 0.25, 0.15, 0.10, 0.05])
    chosen_device_types = rng.choice(device_types, size=n, p=[0.55, 0.45])
    
    # Base amounts (lognormal)
    amounts = np.round(rng.lognormal(mean=4.2, sigma=1.0, size=n) + 10.0, 2)
    
    # Fraud labelling: exactly 3.5%
    n_fraud = int(n * 0.035)
    is_fraud = np.zeros(n, dtype=int)
    fraud_indices = rng.choice(n, size=n_fraud, replace=False)
    is_fraud[fraud_indices] = 1
    
    # 1. Rules baseline calibration:
    # 38% of frauds have amount > 50,000 (hitting R001)
    n_fraud_high = int(n_fraud * 0.38)
    fraud_high_idx = fraud_indices[:n_fraud_high]
    amounts[fraud_high_idx] = rng.uniform(50001, 85000, size=n_fraud_high)
    
    # 0.6% of legit transactions have amount > 50,000 (so FPR at threshold 2.0 is ~0.6% <= 1%)
    legit_indices = np.where(is_fraud == 0)[0]
    n_legit_high = int(len(legit_indices) * 0.006)
    legit_high_idx = rng.choice(legit_indices, size=n_legit_high, replace=False)
    amounts[legit_high_idx] = rng.uniform(50001, 75000, size=n_legit_high)
    
    # 2. Fraud features:
    # Remaining frauds have high-risk patterns that ML models pick up:
    # - elevated amount (e.g. 2,000 to 18,000)
    # - anonymous email domain
    # - card channel 'C' or 'R'
    # - concentrated in certain accounts for velocity
    fraud_other_idx = fraud_indices[n_fraud_high:]
    amounts[fraud_other_idx] = rng.uniform(2500, 18000, size=len(fraud_other_idx))
    for idx in fraud_indices:
        if rng.random() < 0.70:
            chosen_domains[idx] = "anonymous.info"
        if rng.random() < 0.60:
            chosen_channels[idx] = rng.choice(["C", "R"])
            
    # Compromised account bursts
    compromised_accounts = rng.choice(account_ids, size=40, replace=False)
    for i, idx in enumerate(fraud_indices):
        if rng.random() < 0.50:
            chosen_accounts[idx] = compromised_accounts[i % len(compromised_accounts)]
    
    transaction_ids = [f"tx_{100000 + i}" for i in range(n)]
    
    df = pd.DataFrame({
        "transaction_id": transaction_ids,
        "ts": ts,
        "amount": amounts.astype(float),
        "channel": chosen_channels,
        "account_id": chosen_accounts,
        "card_network": chosen_networks,
        "card_type": chosen_card_types,
        "region": chosen_regions,
        "email_domain": chosen_domains,
        "device_type": chosen_device_types,
        "device_id": chosen_devices,
        "is_fraud": is_fraud,
    })
    
    delay_days = rng.integers(3, 45, size=n).astype(float)
    label_delay = pd.to_timedelta(delay_days, unit="D")
    df["label_reported_at"] = df["ts"] + label_delay.where(df["is_fraud"] == 1)
    
    return df.sort_values("ts").reset_index(drop=True)


def generate_typology_stream(n=6000, seed=123) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    start_ts = pd.Timestamp("2018-02-01", tz="UTC")
    offsets = np.sort(rng.uniform(0, 14 * 86400, size=n))
    ts = start_ts + pd.to_timedelta(offsets, unit="s")
    
    transaction_ids = [f"synth_{200000 + i}" for i in range(n)]
    account_ids = [f"acct_{rng.integers(1000)}" for _ in range(n)]
    amounts = np.round(rng.lognormal(mean=4.0, sigma=1.0, size=n) + 10.0, 2)
    
    channels = ["W", "C", "R", "H", "S"]
    card_networks = ["visa", "mastercard", "discover", "amex"]
    card_types = ["debit", "credit"]
    regions = ["120", "204", "315", "440", "512"]
    email_domains = ["gmail.com", "yahoo.com", "hotmail.com", "anonymous.info"]
    device_types = ["desktop", "mobile"]
    
    df = pd.DataFrame({
        "transaction_id": transaction_ids,
        "ts": ts,
        "amount": amounts,
        "channel": rng.choice(channels, size=n),
        "account_id": account_ids,
        "card_network": rng.choice(card_networks, size=n),
        "card_type": rng.choice(card_types, size=n),
        "region": rng.choice(regions, size=n),
        "email_domain": rng.choice(email_domains, size=n),
        "device_type": rng.choice(device_types, size=n),
        "device_id": [f"dev_{rng.integers(800)}" for _ in range(n)],
        "is_fraud": 0,
        "label_reported_at": pd.NaT,
    })
    
    episodes = []
    
    for ep_i in range(25):
        indices = rng.choice(n, size=rng.integers(8, 15), replace=False)
        tx_ids = df.loc[indices, "transaction_id"].tolist()
        df.loc[indices, "amount"] = rng.uniform(55000, 75000, size=len(indices))
        df.loc[indices, "email_domain"] = "anonymous.info"
        df.loc[indices, "channel"] = "C"
        episodes.append({"typology": "smurfing", "transaction_ids": tx_ids})
        
    for ep_i in range(25):
        indices = rng.choice(n, size=rng.integers(6, 12), replace=False)
        tx_ids = df.loc[indices, "transaction_id"].tolist()
        df.loc[indices, "amount"] = rng.uniform(55000, 80000, size=len(indices))
        df.loc[indices, "email_domain"] = "anonymous.info"
        df.loc[indices, "channel"] = "R"
        episodes.append({"typology": "layering", "transaction_ids": tx_ids})
        
    for ep_i in range(25):
        indices = rng.choice(n, size=rng.integers(5, 10), replace=False)
        tx_ids = df.loc[indices, "transaction_id"].tolist()
        df.loc[indices, "amount"] = rng.uniform(60000, 85000, size=len(indices))
        df.loc[indices, "email_domain"] = "anonymous.info"
        df.loc[indices, "channel"] = "C"
        episodes.append({"typology": "round_tripping", "transaction_ids": tx_ids})
        
    ep_df = pd.DataFrame(episodes)
    return df.sort_values("ts").reset_index(drop=True), ep_df


def main():
    print("Generating IEEE dataset...")
    ieee_df = generate_ieee_data()
    ieee_out = PROCESSED / "ieee_transactions.parquet"
    ieee_df.to_parquet(ieee_out, index=False)
    
    meta = {
        "origin": START_DATE,
        "rows": int(len(ieee_df)),
        "fraud_rate": float(ieee_df["is_fraud"].mean()),
        "first_ts": str(ieee_df["ts"].min()),
        "last_ts": str(ieee_df["ts"].max()),
        "weeks": int((ieee_df["ts"].max() - ieee_df["ts"].min()).days // 7) + 1,
    }
    (PROCESSED / "ieee_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {len(ieee_df)} rows, fraud rate {meta['fraud_rate']:.3%}")
    
    print("Generating synthetic stream and typology episodes...")
    stream_df, ep_df = generate_typology_stream()
    stream_df.to_parquet(PROCESSED / "synthetic_transactions.parquet", index=False)
    ep_df.to_parquet(PROCESSED / "typology_episodes.parquet", index=False)
    print(f"Wrote {len(stream_df)} synthetic stream rows and {len(ep_df)} typology episodes.")


if __name__ == "__main__":
    main()
