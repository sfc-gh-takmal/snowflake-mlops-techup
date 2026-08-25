"""Generate synthetic fraud detection dataset and upload to Snowflake.

Produces three tables:
  - CUSTOMER_PROFILES: ~5,000 customers with demographics
  - MERCHANT_DATA: ~500 merchants with risk categories
  - RAW_TRANSACTIONS: ~100,000 transactions with ~3% fraud rate

Fraud patterns injected:
  - High amounts relative to customer history
  - Late-night / early-morning transactions
  - High-risk merchant categories
  - Velocity spikes (many txns in short window)
  - Geographic anomalies (large distance jumps)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "source"))
from config import (
    DATABASE,
    SCHEMA,
    WAREHOUSE,
)
from snowpark_session import create_snowpark_session


def generate_customers(n: int = 5000, rng: np.random.Generator = None) -> pd.DataFrame:
    rng = rng or np.random.default_rng(42)
    return pd.DataFrame(
        {
            "CUSTOMER_ID": [f"CUST_{i:05d}" for i in range(n)],
            "AGE": rng.integers(18, 75, size=n),
            "ANNUAL_INCOME": rng.lognormal(mean=10.8, sigma=0.5, size=n).astype(int),
            "ACCOUNT_AGE_DAYS": rng.integers(30, 3650, size=n),
            "CREDIT_SCORE": np.clip(rng.normal(700, 80, size=n), 300, 850).astype(int),
            "NUM_CARDS": rng.integers(1, 6, size=n),
            "STATE": rng.choice(
                [
                    "CA",
                    "TX",
                    "NY",
                    "FL",
                    "IL",
                    "PA",
                    "OH",
                    "GA",
                    "NC",
                    "MI",
                    "WA",
                    "AZ",
                    "MA",
                    "CO",
                    "VA",
                    "NJ",
                    "OR",
                    "TN",
                    "IN",
                    "MO",
                ],
                size=n,
            ),
        }
    )


def generate_merchants(n: int = 500, rng: np.random.Generator = None) -> pd.DataFrame:
    rng = rng or np.random.default_rng(43)
    categories = [
        "grocery",
        "gas_station",
        "restaurant",
        "online_retail",
        "electronics",
        "travel",
        "entertainment",
        "healthcare",
        "cash_advance",
        "gambling",
        "crypto_exchange",
        "jewelry",
    ]
    risk_map = {
        "grocery": 0.1,
        "gas_station": 0.15,
        "restaurant": 0.12,
        "online_retail": 0.3,
        "electronics": 0.35,
        "travel": 0.25,
        "entertainment": 0.2,
        "healthcare": 0.08,
        "cash_advance": 0.7,
        "gambling": 0.8,
        "crypto_exchange": 0.85,
        "jewelry": 0.5,
    }
    cats = rng.choice(categories, size=n)
    return pd.DataFrame(
        {
            "MERCHANT_ID": [f"MERCH_{i:04d}" for i in range(n)],
            "MERCHANT_NAME": [f"Merchant_{i}" for i in range(n)],
            "CATEGORY": cats,
            "RISK_SCORE": [risk_map[c] + rng.normal(0, 0.05) for c in cats],
            "AVG_TXN_AMOUNT": rng.lognormal(mean=3.5, sigma=1.0, size=n).round(2),
            "CITY": rng.choice(
                [
                    "New York",
                    "Los Angeles",
                    "Chicago",
                    "Houston",
                    "Phoenix",
                    "San Francisco",
                    "Seattle",
                    "Miami",
                    "Denver",
                    "Atlanta",
                ],
                size=n,
            ),
        }
    )


def generate_transactions(
    customers: pd.DataFrame,
    merchants: pd.DataFrame,
    n: int = 100_000,
    fraud_rate: float = 0.03,
    rng: np.random.Generator = None,
) -> pd.DataFrame:
    rng = rng or np.random.default_rng(44)

    n_fraud = int(n * fraud_rate)
    n_legit = n - n_fraud

    customer_ids = customers["CUSTOMER_ID"].values
    merchant_ids = merchants["MERCHANT_ID"].values

    # Base timestamp: 90 days of data
    base_time = pd.Timestamp("2024-01-01")
    end_time = pd.Timestamp("2024-03-31")
    time_range_seconds = int((end_time - base_time).total_seconds())

    # --- Legitimate transactions ---
    legit_customers = rng.choice(customer_ids, size=n_legit)
    legit_merchants = rng.choice(merchant_ids, size=n_legit)
    legit_amounts = rng.lognormal(mean=3.2, sigma=1.0, size=n_legit).round(2)
    legit_amounts = np.clip(legit_amounts, 1.0, 5000.0)
    legit_times = base_time + pd.to_timedelta(rng.integers(0, time_range_seconds, size=n_legit), unit="s")
    # Mostly daytime, but a real minority of legitimate activity happens at night.
    # Drawing legit from 6-22 and fraud from {0-5,23} (the previous behaviour) made
    # the classes perfectly separable by hour: IS_LATE_NIGHT alone gave AUC 1.000,
    # so the model learned a synthetic artefact instead of a fraud pattern.
    legit_hours = rng.choice(np.arange(24), size=n_legit, p=_hour_probs(night_weight=0.12))

    # --- Fraudulent transactions (with injected patterns) ---
    fraud_customers = rng.choice(customer_ids, size=n_fraud)
    # Fraud skews to risky merchants but is not exclusive to them, so
    # IS_HIGH_RISK_MERCHANT is a signal rather than a giveaway.
    high_risk = merchants[merchants["RISK_SCORE"] > 0.4]["MERCHANT_ID"].values
    n_fraud_high_risk = int(n_fraud * 0.75)
    fraud_merchants = np.concatenate(
        [
            rng.choice(high_risk, size=n_fraud_high_risk),
            rng.choice(merchant_ids, size=n_fraud - n_fraud_high_risk),
        ]
    )
    rng.shuffle(fraud_merchants)
    # Higher amounts for fraud, but the distributions overlap
    fraud_amounts = rng.lognormal(mean=4.6, sigma=1.2, size=n_fraud).round(2)
    fraud_amounts = np.clip(fraud_amounts, 5.0, 25000.0)
    fraud_times = base_time + pd.to_timedelta(rng.integers(0, time_range_seconds, size=n_fraud), unit="s")
    # Fraud is ~5x more likely at night, but plenty of fraud happens in daylight.
    fraud_hours = rng.choice(np.arange(24), size=n_fraud, p=_hour_probs(night_weight=0.55))

    # Combine
    all_customers = np.concatenate([legit_customers, fraud_customers])
    all_merchants = np.concatenate([legit_merchants, fraud_merchants])
    all_amounts = np.concatenate([legit_amounts, fraud_amounts])
    all_times = np.concatenate([legit_times, fraud_times])
    all_hours = np.concatenate([legit_hours, fraud_hours])
    is_fraud = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)]).astype(int)

    # Adjust timestamps to reflect actual hour
    all_times = pd.to_datetime(all_times)
    all_times = all_times.map(lambda t: t.replace(hour=0, minute=0, second=0)) + pd.to_timedelta(
        all_hours * 3600 + rng.integers(0, 3600, size=n), unit="s"
    )

    # Device and location
    devices = rng.choice(["mobile", "desktop", "tablet", "pos_terminal"], size=n, p=[0.45, 0.25, 0.1, 0.2])
    lats = rng.uniform(25.0, 48.0, size=n).round(4)
    lons = rng.uniform(-125.0, -70.0, size=n).round(4)

    # Shuffle
    idx = rng.permutation(n)

    df = pd.DataFrame(
        {
            "TXN_ID": [f"TXN_{i:07d}" for i in range(n)],
            "CUSTOMER_ID": all_customers[idx],
            "MERCHANT_ID": all_merchants[idx],
            "AMOUNT": all_amounts[idx],
            "TIMESTAMP": all_times[idx],
            "IS_FRAUD": is_fraud[idx],
            "DEVICE_TYPE": devices[idx],
            "LOCATION_LAT": lats[idx],
            "LOCATION_LON": lons[idx],
        }
    )

    return df.sort_values("TIMESTAMP").reset_index(drop=True)


def _hour_probs(night_weight: float = 0.12):
    """Hour-of-day distribution over all 24 hours.

    `night_weight` is the total probability mass assigned to the "late night" window
    (hours 23, 0-5) that IS_LATE_NIGHT flags; the remainder is spread over daytime
    hours with a realistic mid-afternoon peak.

    Both classes draw from all 24 hours so the hour feature is predictive without
    being deterministic. Legit uses a low night_weight, fraud a high one.
    """
    night_hours = [23, 0, 1, 2, 3, 4, 5]
    day_hours = list(range(6, 23))
    day_weights = [1, 2, 4, 6, 7, 8, 9, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2]

    probs = [0.0] * 24
    # Night mass: heaviest in the small hours
    night_shape = [3, 4, 4, 3, 2, 2, 2]
    night_total = sum(night_shape)
    for h, w in zip(night_hours, night_shape):
        probs[h] = night_weight * w / night_total
    # Day mass gets whatever is left
    day_total = sum(day_weights)
    for h, w in zip(day_hours, day_weights):
        probs[h] = (1.0 - night_weight) * w / day_total

    total = sum(probs)
    return [p / total for p in probs]


def upload_to_snowflake(
    session, df: pd.DataFrame, table_name: str, database: str | None = None, schema: str | None = None
):
    """Write pandas DataFrame to Snowflake table."""
    db = database or DATABASE
    sch = schema or SCHEMA

    session.sql(f"USE WAREHOUSE {WAREHOUSE}").collect()

    # Convert datetime columns to string for proper TIMESTAMP handling
    df_copy = df.copy()
    for col in df_copy.select_dtypes(include=["datetime64"]).columns:
        df_copy[col] = df_copy[col].dt.strftime("%Y-%m-%d %H:%M:%S")

    tbl_short = table_name.split(".")[-1]
    session.write_pandas(df_copy, tbl_short, database=db, schema=sch, auto_create_table=True, overwrite=True)

    # Fix TIMESTAMP columns (write_pandas stores datetime-as-string as VARCHAR)
    fqn = f"{db}.{sch}.{tbl_short}"
    if "TIMESTAMP" in df.columns:
        session.sql(f"""
            CREATE OR REPLACE TABLE {fqn} AS
            SELECT * EXCLUDE (TIMESTAMP),
                   TO_TIMESTAMP_NTZ(TIMESTAMP) AS TIMESTAMP
            FROM {fqn}
        """).collect()

    count = session.table(fqn).count()
    print(f"  {fqn}: {count:,} rows")


def main():
    print("Generating synthetic fraud detection dataset...")
    rng = np.random.default_rng(42)

    customers = generate_customers(n=5000, rng=rng)
    print(f"  Customers: {len(customers):,}")

    merchants = generate_merchants(n=500, rng=rng)
    print(f"  Merchants: {len(merchants):,}")

    transactions = generate_transactions(customers, merchants, n=100_000, rng=rng)
    print(f"  Transactions: {len(transactions):,} ({transactions['IS_FRAUD'].mean():.1%} fraud)")

    print("\nConnecting to Snowflake...")
    session = create_snowpark_session()

    # Upload to both DEV and PROD environments
    environments = [
        ("SNOW_MLOPS_DEV", "ML"),
        ("SNOW_MLOPS_PROD", "ML"),
    ]

    for db, schema in environments:
        print(f"\nUploading to {db}.{schema}...")
        session.sql(f"USE DATABASE {db}").collect()
        session.sql(f"USE SCHEMA {schema}").collect()

        upload_to_snowflake(session, customers, f"{db}.{schema}.CUSTOMER_PROFILES", database=db, schema=schema)
        upload_to_snowflake(session, merchants, f"{db}.{schema}.MERCHANT_DATA", database=db, schema=schema)
        upload_to_snowflake(session, transactions, f"{db}.{schema}.RAW_TRANSACTIONS", database=db, schema=schema)

    print("\nDataset generation complete (DEV + PROD)!")
    session.close()


if __name__ == "__main__":
    main()
