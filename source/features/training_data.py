"""Generate training dataset from Feature Store with point-in-time correctness."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import snowflake.snowpark.functions as F
from config import DATABASE, SCHEMA, SOURCE_DATABASE, SOURCE_SCHEMA, WAREHOUSE
from snowflake.ml.feature_store import CreationMode, FeatureStore
from snowpark_session import create_snowpark_session


def generate_training_data(session=None):
    """Generate training dataset by joining Feature Store features with labels.

    Returns a Snowpark DataFrame with features + IS_FRAUD label.
    """
    close_session = False
    if session is None:
        session = create_snowpark_session()
        close_session = True

    session.sql(f"USE WAREHOUSE {WAREHOUSE}").collect()

    fs = FeatureStore(
        session=session,
        database=DATABASE,
        name=SCHEMA,
        default_warehouse=WAREHOUSE,
        creation_mode=CreationMode.FAIL_IF_NOT_EXIST,
    )

    # Get feature views
    cust_fv = fs.get_feature_view("CUSTOMER_RISK_FEATURES", "V1")

    # Build spine from source tables (PROD) -- labels + entity keys
    spine = session.table(f"{SOURCE_DATABASE}.{SOURCE_SCHEMA}.RAW_TRANSACTIONS").select(
        F.col("TXN_ID"),
        F.col("CUSTOMER_ID"),
        F.col("TIMESTAMP"),
        F.col("IS_FRAUD"),
    )

    # Generate dataset - use only transaction FV (which contains customer features too)
    # The txn FV already joins merchant data and customer averages inline
    training_dataset = fs.generate_dataset(
        name="FRAUD_TRAINING_DATA",
        version="V1",
        spine_df=spine,
        features=[cust_fv],
        spine_timestamp_col="TIMESTAMP",
        spine_label_cols=["IS_FRAUD"],
        desc="Training dataset for fraud detection model",
    )

    print(f"Training dataset generated: {training_dataset.read.to_snowpark_dataframe().count():,} rows")

    if close_session:
        session.close()

    return training_dataset


if __name__ == "__main__":
    generate_training_data()
