"""Batch Inference CI entry point.

Scores the feature view table using the latest registered model and validates
predictions are sane. Used in both STAGE and PROD CI workflows.

Exit code 1 if validation fails (null predictions or degenerate probabilities).
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "source"))
from config import MODEL_NAME, SCORING_VIEW_NAME
from snowpark_session import create_snowpark_session

# Environment-aware configuration
DATABASE = os.getenv("SNOWFLAKE_DATABASE", "SNOW_MLOPS_DEV")
SCHEMA = os.getenv("SNOWFLAKE_SCHEMA", "ML")
WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE", f"{DATABASE}_WH")

# Score the scoring view (one row per transaction) rather than the raw customer
# feature table, so batch inference uses the same feature contract as training.
INPUT_TABLE = f"{DATABASE}.{SCHEMA}.{SCORING_VIEW_NAME}"
OUTPUT_TABLE = f"{DATABASE}.{SCHEMA}.BATCH_PREDICTIONS"


def write_batch_summary(result: dict, validation: dict):
    """Write batch inference results to GitHub Actions Job Summary."""
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    status = "PASSED" if validation["passed"] else "FAILED"
    md = f"""## Batch Inference — {status}

| Metric | Value |
|--------|-------|
| Rows scored | {result["row_count"]:,} |
| Model version | `{result["model_version"]}` |
| Null predictions | {validation["null_predictions"]} |
| Avg probability sum | {validation["avg_probability_sum"]:.6f} |
| Max deviation from 1.0 | {validation["max_deviation_from_1"]:.6f} |
| Output table | `{result["output_table"]}` |
"""
    with open(summary_path, "a") as f:
        f.write(md)
    print("  Batch summary written to GITHUB_STEP_SUMMARY")


def main():
    print("=" * 60)
    print("BATCH INFERENCE")
    print("=" * 60)

    session = create_snowpark_session()
    session.sql(f"USE DATABASE {DATABASE}").collect()
    session.sql(f"USE SCHEMA {SCHEMA}").collect()
    session.sql(f"USE WAREHOUSE {WAREHOUSE}").collect()

    # Ensure Feature Views + scoring view exist in this environment (idempotent)
    print(f"\n[0/2] Ensuring Feature Views exist in {DATABASE}.{SCHEMA}...")
    from features.feature_views import register_feature_views, register_scoring_view

    register_feature_views(session=session, database=DATABASE, schema=SCHEMA, warehouse=WAREHOUSE)
    register_scoring_view(session=session, database=DATABASE, schema=SCHEMA)
    print("  Feature Views + scoring view registered (idempotent).")

    from serving.batch_inference import run_batch_inference, validate_predictions

    # Step 1: Run batch scoring
    print(f"\n[1/2] Scoring {INPUT_TABLE}...")
    result = run_batch_inference(
        session=session,
        input_table=INPUT_TABLE,
        output_table=OUTPUT_TABLE,
        model_name=MODEL_NAME,
    )

    # Step 2: Validate predictions
    print("\n[2/2] Validating predictions...")
    validation = validate_predictions(session, OUTPUT_TABLE)

    if validation["passed"]:
        print("  Validation PASSED")
        print(f"    Rows: {validation['total_rows']:,}")
        print(f"    Null predictions: {validation['null_predictions']}")
        print(f"    Max deviation: {validation['max_deviation_from_1']:.6f}")
    else:
        print("  Validation FAILED")
        print(f"    Null predictions: {validation['null_predictions']}")
        print(f"    Max deviation: {validation['max_deviation_from_1']:.6f}")

    # Write summary for CI
    write_batch_summary(result, validation)

    print("\n" + "=" * 60)
    print(f"BATCH INFERENCE COMPLETE — {result['row_count']:,} rows scored")
    print(f"  Output: {OUTPUT_TABLE}")
    print(f"  Model: {MODEL_NAME} ({result['model_version']})")
    print("=" * 60)

    session.close()

    if not validation["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
