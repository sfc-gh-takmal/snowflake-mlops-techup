"""Define and register Feature Views for fraud detection.

Feature Views:
  1. CUSTOMER_RISK_FEATURES - Aggregated customer behavior signals
  2. TRANSACTION_CONTEXT_FEATURES - Per-transaction contextual signals
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import snowflake.snowpark.functions as F
from config import (
    DATABASE,
    FEATURE_COLUMNS,
    FEATURE_VIEW_CONFIG,
    FEATURE_VIEW_NAME,
    FEATURE_VIEW_VERSION,
    SCHEMA,
    SCORING_VIEW_NAME,
    SOURCE_DATABASE,
    SOURCE_SCHEMA,
    TRANSACTION_FEATURE_COLUMNS,
    TXN_FEATURE_VIEW_NAME,
    TXN_FEATURE_VIEW_VERSION,
    WAREHOUSE,
)
from snowflake.ml.feature_store import CreationMode, Entity, FeatureStore, FeatureView
from snowflake.snowpark import Session
from snowpark_session import create_snowpark_session


def create_customer_features_df(session: Session):
    """Build Snowpark DataFrame for customer-level risk features."""
    txn = session.table(f"{SOURCE_DATABASE}.{SOURCE_SCHEMA}.RAW_TRANSACTIONS")
    cust = session.table(f"{SOURCE_DATABASE}.{SOURCE_SCHEMA}.CUSTOMER_PROFILES")

    # Aggregate features per customer (latest snapshot)
    customer_agg = txn.group_by("CUSTOMER_ID").agg(
        F.count("*").alias("TOTAL_TXN_COUNT"),
        F.avg("AMOUNT").alias("AVG_TXN_AMOUNT"),
        F.max("AMOUNT").alias("MAX_TXN_AMOUNT"),
        F.stddev("AMOUNT").alias("STDDEV_TXN_AMOUNT"),
        F.count_distinct("MERCHANT_ID").alias("UNIQUE_MERCHANTS"),
        F.sum(F.when(F.col("IS_FRAUD") == 1, F.lit(1)).otherwise(F.lit(0))).alias("HISTORICAL_FRAUD_COUNT"),
        F.avg(F.col("IS_FRAUD").cast("float")).alias("HISTORICAL_FRAUD_RATE"),
        F.count_distinct(F.dayofyear("TIMESTAMP")).alias("ACTIVE_DAYS"),
        F.avg(
            F.when(F.hour("TIMESTAMP") < 6, F.lit(1)).when(F.hour("TIMESTAMP") > 22, F.lit(1)).otherwise(F.lit(0))
        ).alias("LATE_NIGHT_TXN_RATIO"),
        F.max("TIMESTAMP").alias("FEATURE_TS"),
    )

    # Join with customer profile for enrichment
    features = customer_agg.join(cust, on="CUSTOMER_ID", how="inner").select(
        F.col("CUSTOMER_ID"),
        F.col("TOTAL_TXN_COUNT"),
        F.col("AVG_TXN_AMOUNT"),
        F.col("MAX_TXN_AMOUNT"),
        F.col("STDDEV_TXN_AMOUNT"),
        F.col("UNIQUE_MERCHANTS"),
        F.col("HISTORICAL_FRAUD_COUNT"),
        F.col("HISTORICAL_FRAUD_RATE"),
        F.col("ACTIVE_DAYS"),
        F.col("LATE_NIGHT_TXN_RATIO"),
        F.col("CREDIT_SCORE"),
        F.col("ACCOUNT_AGE_DAYS"),
        F.col("ANNUAL_INCOME"),
        F.col("FEATURE_TS"),
    )
    return features


def create_transaction_features_df(session: Session):
    """Build Snowpark DataFrame for transaction-level context features."""
    txn = session.table(f"{SOURCE_DATABASE}.{SOURCE_SCHEMA}.RAW_TRANSACTIONS")
    merch = session.table(f"{SOURCE_DATABASE}.{SOURCE_SCHEMA}.MERCHANT_DATA")
    cust_avg = txn.group_by("CUSTOMER_ID").agg(F.avg("AMOUNT").alias("CUST_AVG_AMOUNT"))

    features = (
        txn.join(merch, on="MERCHANT_ID", how="left")
        .join(cust_avg, on="CUSTOMER_ID", how="left")
        .select(
            F.col("TXN_ID"),
            F.col("CUSTOMER_ID"),
            F.col("AMOUNT"),
            (F.col("AMOUNT") / F.coalesce(F.col("CUST_AVG_AMOUNT"), F.lit(1.0))).alias("AMOUNT_TO_AVG_RATIO"),
            F.when(F.col("RISK_SCORE") > 0.5, F.lit(1)).otherwise(F.lit(0)).alias("IS_HIGH_RISK_MERCHANT"),
            F.col("RISK_SCORE").alias("MERCHANT_RISK_SCORE"),
            F.hour("TIMESTAMP").alias("HOUR_OF_DAY"),
            F.when(F.dayofweek("TIMESTAMP").isin([0, 6]), F.lit(1)).otherwise(F.lit(0)).alias("IS_WEEKEND"),
            F.when((F.hour("TIMESTAMP") < 6) | (F.hour("TIMESTAMP") > 22), F.lit(1))
            .otherwise(F.lit(0))
            .alias("IS_LATE_NIGHT"),
            F.col("DEVICE_TYPE"),
            F.col("CATEGORY").alias("MERCHANT_CATEGORY"),
            F.col("TIMESTAMP").alias("FEATURE_TS"),
        )
    )
    return features


def register_feature_views(session=None, database=None, schema=None, warehouse=None):
    """Register all feature views in the Feature Store. Idempotent (creates or updates)."""
    close_session = False
    if session is None:
        session = create_snowpark_session()
        close_session = True

    db = database or DATABASE
    sch = schema or SCHEMA
    wh = warehouse or WAREHOUSE

    session.sql(f"USE WAREHOUSE {wh}").collect()

    fs = FeatureStore(
        session=session,
        database=db,
        name=sch,
        default_warehouse=wh,
        creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
    )

    customer_entity = Entity(name="CUSTOMER", join_keys=["CUSTOMER_ID"])
    transaction_entity = Entity(name="TRANSACTION", join_keys=["TXN_ID"])

    # Register entities first
    fs.register_entity(customer_entity)
    fs.register_entity(transaction_entity)

    # Customer risk features
    print("Creating CUSTOMER_RISK_FEATURES feature view...")
    cust_df = create_customer_features_df(session)
    customer_fv = FeatureView(
        name="CUSTOMER_RISK_FEATURES",
        entities=[customer_entity],
        feature_df=cust_df,
        timestamp_col="FEATURE_TS",
        refresh_freq=FEATURE_VIEW_CONFIG.get("customer_features_refresh", "1 hour"),
        desc="Customer-level risk signals for fraud detection",
    )
    customer_fv = fs.register_feature_view(feature_view=customer_fv, version="V1", overwrite=True)
    print(
        f"  Registered: CUSTOMER_RISK_FEATURES/V1 (refresh={FEATURE_VIEW_CONFIG.get('customer_features_refresh', '1 hour')})"
    )

    # Transaction context features
    print("Creating TRANSACTION_CONTEXT_FEATURES feature view...")
    txn_df = create_transaction_features_df(session)
    txn_fv = FeatureView(
        name="TRANSACTION_CONTEXT_FEATURES",
        entities=[transaction_entity],
        feature_df=txn_df,
        timestamp_col="FEATURE_TS",
        refresh_freq=FEATURE_VIEW_CONFIG.get("transaction_features_refresh", "1 hour"),
        desc="Per-transaction contextual signals for fraud detection",
    )
    txn_fv = fs.register_feature_view(feature_view=txn_fv, version="V1", overwrite=True)
    print(
        f"  Registered: TRANSACTION_CONTEXT_FEATURES/V1 (refresh={FEATURE_VIEW_CONFIG.get('transaction_features_refresh', '1 hour')})"
    )

    if close_session:
        session.close()

    return fs, customer_fv, txn_fv


def register_scoring_view(session=None, database=None, schema=None):
    """Create the scoring view: one row per transaction, features from both views.

    Training, batch inference and the real-time service all read this view, so the
    feature contract is defined in exactly one place (config.FEATURE_COLUMNS).

    The join is at TXN_ID granularity on purpose. Joining customer-level features
    straight onto RAW_TRANSACTIONS produces ~20 rows per customer sharing one
    feature vector but carrying different labels, which makes the target
    unlearnable and pins precision to the base rate.
    """
    close_session = False
    if session is None:
        session = create_snowpark_session()
        close_session = True

    db = database or DATABASE
    sch = schema or SCHEMA

    txn_cols = set(TRANSACTION_FEATURE_COLUMNS)
    feature_select = ",\n        ".join(f"t.{c}" if c in txn_cols else f"c.{c}" for c in FEATURE_COLUMNS)

    session.sql(f"""
        CREATE OR REPLACE VIEW {db}.{sch}.{SCORING_VIEW_NAME} AS
        SELECT
            t.TXN_ID,
            t.CUSTOMER_ID,
            {feature_select},
            t.FEATURE_TS
        FROM {db}.{sch}."{TXN_FEATURE_VIEW_NAME}${TXN_FEATURE_VIEW_VERSION}" t
        JOIN {db}.{sch}."{FEATURE_VIEW_NAME}${FEATURE_VIEW_VERSION}" c
          ON t.CUSTOMER_ID = c.CUSTOMER_ID
    """).collect()

    checks = session.sql(f"""
        SELECT COUNT(*) AS N, COUNT(DISTINCT TXN_ID) AS N_TXN
        FROM {db}.{sch}.{SCORING_VIEW_NAME}
    """).collect()[0]
    if checks["N"] != checks["N_TXN"]:
        raise RuntimeError(
            f"{SCORING_VIEW_NAME} has {checks['N']} rows for {checks['N_TXN']} transactions -- join fans out"
        )
    print(f"  Registered scoring view: {SCORING_VIEW_NAME} ({checks['N']:,} rows, 1 per transaction)")

    if close_session:
        session.close()

    return SCORING_VIEW_NAME


if __name__ == "__main__":
    register_feature_views()
    register_scoring_view()
