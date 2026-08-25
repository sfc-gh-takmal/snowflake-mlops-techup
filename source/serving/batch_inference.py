"""Batch inference: score a table using the registered model on warehouse compute.

Uses model.run() from the Model Registry — no SPCS service needed.
Runs on the warehouse, suitable for bulk scoring (daily/hourly).
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATABASE, MODEL_NAME, SCHEMA


def run_batch_inference(
    session,
    input_table: str,
    output_table: str,
    model_name: str = MODEL_NAME,
    version: str | None = None,
) -> dict:
    """Score an entire table using the registered model.

    Args:
        session: Active Snowpark session
        input_table: Fully-qualified input table with feature columns
        output_table: Where to write predictions (overwritten each run)
        model_name: Model to use from the registry
        version: Specific version, or None for default

    Returns:
        dict with row_count and output_table
    """
    from snowflake.ml.registry import Registry

    db = os.getenv("SNOWFLAKE_DATABASE", DATABASE)
    schema = os.getenv("SNOWFLAKE_SCHEMA", SCHEMA)

    reg = Registry(session=session, database_name=db, schema_name=schema)
    model = reg.get_model(model_name)
    mv = model.version(version) if version else model.default

    print(f"  Model: {model_name} (version: {mv.version_name})")
    print(f"  Input: {input_table}")

    input_df = session.table(input_table)
    row_count = input_df.count()
    print(f"  Rows to score: {row_count:,}")

    predictions_df = mv.run(input_df, function_name="predict_proba")

    # Add timestamp for model monitoring.
    # ML Observability requires TIMESTAMP_NTZ: current_timestamp() returns
    # TIMESTAMP_LTZ, which CREATE MODEL MONITOR rejects. Cast explicitly.
    from snowflake.snowpark.functions import current_timestamp
    from snowflake.snowpark.types import TimestampTimeZone, TimestampType

    predictions_df = predictions_df.with_column(
        "PREDICTION_TS",
        current_timestamp().cast(TimestampType(TimestampTimeZone.NTZ)),
    )

    predictions_df.write.mode("overwrite").save_as_table(output_table)
    print(f"  Output written to: {output_table}")

    # A model monitor materializes SOURCE into dynamic tables, which requires
    # change tracking on the base table. save_as_table(mode="overwrite")
    # recreates the table and drops the setting, so re-apply it every run.
    session.sql(f"ALTER TABLE {output_table} SET CHANGE_TRACKING = TRUE").collect()
    print("  Change tracking enabled (required by ML Observability)")

    return {
        "row_count": row_count,
        "output_table": output_table,
        "model_version": mv.version_name,
    }


def validate_predictions(session, output_table: str) -> dict:
    """Validate batch predictions are sane (no nulls, probabilities sum to ~1)."""
    df = session.table(output_table)
    total = df.count()

    # Column names from model.run() are lowercase quoted identifiers
    col_0 = '"output_feature_0"'
    col_1 = '"output_feature_1"'

    # Check for null predictions
    null_count = df.filter(f"{col_0} IS NULL OR {col_1} IS NULL").count()

    # Check probability sums (should be ~1.0)
    stats = df.select(col_0, col_1).to_pandas()

    prob_sums = stats["output_feature_0"] + stats["output_feature_1"]
    avg_sum = prob_sums.mean()
    max_deviation = (prob_sums - 1.0).abs().max()

    validation = {
        "total_rows": total,
        "null_predictions": null_count,
        "avg_probability_sum": float(avg_sum),
        "max_deviation_from_1": float(max_deviation),
        "passed": null_count == 0 and max_deviation < 0.01,
    }

    return validation
