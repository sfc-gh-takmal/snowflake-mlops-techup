"""ML Pipeline Task DAG — Python SDK with configurable compute per step.

Uses snowflake.core.task.dagv1 (DAG, DAGTask, DAGOperation) for task management
and snowflake.ml.jobs.remote for ML Job definitions on compute pools.

Each step can run on either:
  - Warehouse: StoredProcedureCall (for SQL-heavy / lightweight Python)
  - SPCS Compute Pool: @remote ML Job (for training, GPU, custom packages)

Configured via PIPELINE_CONFIG in source/config.py:
  "feature_engineering_compute": "warehouse" | "spcs"
  "training_compute": "warehouse" | "spcs"
  "evaluation_compute": "warehouse" | "spcs"

Usage:
    python source/pipeline/ml_pipeline_dag.py --deploy --env stage
    python source/pipeline/ml_pipeline_dag.py --execute --env stage
    python source/pipeline/ml_pipeline_dag.py --status --env stage
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    EXPERIMENT_CONFIG,
    FEATURE_COLUMNS,
    FEATURE_VIEW_CONFIG,
    LEARNING_RATE,
    MAX_DEPTH,
    N_ESTIMATORS,
    PIPELINE_CONFIG,
    RETRAIN_CONFIG,
    SCALE_POS_WEIGHT,
    SCORING_VIEW_NAME,
    TRANSACTION_FEATURE_COLUMNS,
)
from snowflake.core import Root
from snowflake.core._common import CreateMode
from snowflake.core.task import StoredProcedureCall
from snowflake.core.task.dagv1 import DAG, DAGOperation, DAGTask
from snowflake.ml.jobs import remote
from snowflake.snowpark import Session
from snowpark_session import create_snowpark_session

# Environment configurations
ENV_CONFIG = {
    "dev": {
        "database": "SNOW_MLOPS_DEV",
        "schema": "ML",
        "warehouse": "SNOW_MLOPS_DEV_WH",
        "compute_pool": "SNOW_MLOPS_DEV_POOL",
        "source_database": "SNOW_MLOPS_PROD",
        "source_schema": "ML",
    },
    "stage": {
        "database": "SNOW_MLOPS_STAGE",
        "schema": "ML",
        "warehouse": "SNOW_MLOPS_STAGE_WH",
        "compute_pool": "SNOW_MLOPS_STAGE_POOL",
        "source_database": "SNOW_MLOPS_PROD",
        "source_schema": "ML",
    },
}

DAG_NAME = "ML_TRAINING_PIPELINE"


def get_env_config(env: str) -> dict:
    if env not in ENV_CONFIG:
        raise ValueError(f"Unknown environment: {env}. Use 'dev' or 'stage'.")
    return ENV_CONFIG[env]


# ─── ML Job Definitions (@remote) ────────────────────────────────────────────
# These run on compute pools when the corresponding config is set to "spcs"


def build_feature_eng_remote(cfg: dict):
    """Build the @remote-decorated feature engineering function."""
    pool = cfg["compute_pool"]
    db = cfg["database"]
    schema = cfg["schema"]
    wh = cfg["warehouse"]
    src_db = cfg["source_database"]
    src_schema = cfg["source_schema"]
    stage = f"@{db}.{schema}.PIPELINE_STAGE"
    cust_refresh = FEATURE_VIEW_CONFIG.get("customer_features_refresh", "1 hour")
    txn_refresh = FEATURE_VIEW_CONFIG.get("transaction_features_refresh", "1 hour")
    scoring_view = SCORING_VIEW_NAME
    feature_cols = list(FEATURE_COLUMNS)
    txn_cols = set(TRANSACTION_FEATURE_COLUMNS)

    @remote(
        pool,
        stage_name=stage,
        pip_requirements=["snowflake-ml-python"],
    )
    def feature_engineering() -> str:
        # All imports must be inside the function body -- cloudpickle does not
        # reliably serialize module references captured from the outer scope.
        import json

        from snowflake.ml.feature_store import CreationMode, Entity, FeatureStore, FeatureView
        from snowflake.snowpark import Session as _Session

        session = _Session.builder.getOrCreate()
        session.sql(f"USE WAREHOUSE {wh}").collect()

        fs = FeatureStore(
            session=session,
            database=db,
            name=schema,
            default_warehouse=wh,
            creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
        )

        customer_entity = Entity(name="CUSTOMER", join_keys=["CUSTOMER_ID"])
        transaction_entity = Entity(name="TRANSACTION", join_keys=["TXN_ID"])
        fs.register_entity(customer_entity)
        fs.register_entity(transaction_entity)

        # Feature SQL is inlined rather than imported from source/features so the
        # job is self-contained on the compute pool. Keep in sync with
        # source/features/feature_views.py, which the warehouse path uses.
        cust_df = session.sql(f"""
            SELECT
                a.CUSTOMER_ID,
                a.TOTAL_TXN_COUNT, a.AVG_TXN_AMOUNT, a.MAX_TXN_AMOUNT, a.STDDEV_TXN_AMOUNT,
                a.UNIQUE_MERCHANTS, a.HISTORICAL_FRAUD_COUNT, a.HISTORICAL_FRAUD_RATE,
                a.ACTIVE_DAYS, a.LATE_NIGHT_TXN_RATIO,
                p.CREDIT_SCORE, p.ACCOUNT_AGE_DAYS, p.ANNUAL_INCOME,
                a.FEATURE_TS
            FROM (
                SELECT
                    CUSTOMER_ID,
                    COUNT(*)                                        AS TOTAL_TXN_COUNT,
                    AVG(AMOUNT)                                     AS AVG_TXN_AMOUNT,
                    MAX(AMOUNT)                                     AS MAX_TXN_AMOUNT,
                    STDDEV(AMOUNT)                                  AS STDDEV_TXN_AMOUNT,
                    COUNT(DISTINCT MERCHANT_ID)                     AS UNIQUE_MERCHANTS,
                    SUM(IFF(IS_FRAUD = 1, 1, 0))                    AS HISTORICAL_FRAUD_COUNT,
                    AVG(IS_FRAUD::FLOAT)                            AS HISTORICAL_FRAUD_RATE,
                    COUNT(DISTINCT DAYOFYEAR(TIMESTAMP))            AS ACTIVE_DAYS,
                    AVG(IFF(HOUR(TIMESTAMP) < 6
                            OR HOUR(TIMESTAMP) > 22, 1, 0))         AS LATE_NIGHT_TXN_RATIO,
                    MAX(TIMESTAMP)                                  AS FEATURE_TS
                FROM {src_db}.{src_schema}.RAW_TRANSACTIONS
                GROUP BY CUSTOMER_ID
            ) a
            JOIN {src_db}.{src_schema}.CUSTOMER_PROFILES p
              ON a.CUSTOMER_ID = p.CUSTOMER_ID
        """)

        customer_fv = FeatureView(
            name="CUSTOMER_RISK_FEATURES",
            entities=[customer_entity],
            feature_df=cust_df,
            timestamp_col="FEATURE_TS",
            refresh_freq=cust_refresh,
            desc="Customer-level risk signals for fraud detection",
        )
        fs.register_feature_view(feature_view=customer_fv, version="V1", overwrite=True)

        txn_df = session.sql(f"""
            SELECT
                t.TXN_ID,
                t.CUSTOMER_ID,
                t.AMOUNT,
                t.AMOUNT / COALESCE(c.CUST_AVG_AMOUNT, 1.0)      AS AMOUNT_TO_AVG_RATIO,
                IFF(m.RISK_SCORE > 0.5, 1, 0)                    AS IS_HIGH_RISK_MERCHANT,
                m.RISK_SCORE                                     AS MERCHANT_RISK_SCORE,
                HOUR(t.TIMESTAMP)                                AS HOUR_OF_DAY,
                IFF(DAYOFWEEK(t.TIMESTAMP) IN (0, 6), 1, 0)      AS IS_WEEKEND,
                IFF(HOUR(t.TIMESTAMP) < 6
                    OR HOUR(t.TIMESTAMP) > 22, 1, 0)             AS IS_LATE_NIGHT,
                t.DEVICE_TYPE,
                m.CATEGORY                                       AS MERCHANT_CATEGORY,
                t.TIMESTAMP                                      AS FEATURE_TS
            FROM {src_db}.{src_schema}.RAW_TRANSACTIONS t
            LEFT JOIN {src_db}.{src_schema}.MERCHANT_DATA m
              ON t.MERCHANT_ID = m.MERCHANT_ID
            LEFT JOIN (
                SELECT CUSTOMER_ID, AVG(AMOUNT) AS CUST_AVG_AMOUNT
                FROM {src_db}.{src_schema}.RAW_TRANSACTIONS
                GROUP BY CUSTOMER_ID
            ) c ON t.CUSTOMER_ID = c.CUSTOMER_ID
        """)

        txn_fv = FeatureView(
            name="TRANSACTION_CONTEXT_FEATURES",
            entities=[transaction_entity],
            feature_df=txn_df,
            timestamp_col="FEATURE_TS",
            refresh_freq=txn_refresh,
            desc="Per-transaction contextual signals for fraud detection",
        )
        fs.register_feature_view(feature_view=txn_fv, version="V1", overwrite=True)

        # Scoring view: one row per transaction, combining transaction context with
        # customer history. Training, batch inference and the SPCS service all read
        # this, so the feature contract cannot drift between them.
        #
        # Note the join is on TXN_ID granularity. Joining customer-level features
        # directly to RAW_TRANSACTIONS (the previous approach) produced 20 rows per
        # customer with identical features but differing labels, which made the
        # target unlearnable and pinned precision to the base rate.
        feature_select = ",\n                ".join(f"t.{c}" if c in txn_cols else f"c.{c}" for c in feature_cols)
        session.sql(f"""
            CREATE OR REPLACE VIEW {db}.{schema}.{scoring_view} AS
            SELECT
                t.TXN_ID,
                t.CUSTOMER_ID,
                {feature_select},
                t.FEATURE_TS
            FROM {db}.{schema}."TRANSACTION_CONTEXT_FEATURES$V1" t
            JOIN {db}.{schema}."CUSTOMER_RISK_FEATURES$V1" c
              ON t.CUSTOMER_ID = c.CUSTOMER_ID
        """).collect()

        checks = session.sql(f"""
            SELECT COUNT(*) AS N, COUNT(DISTINCT TXN_ID) AS N_TXN
            FROM {db}.{schema}.{scoring_view}
        """).collect()[0]
        if checks["N"] == 0:
            raise RuntimeError(f"{scoring_view} is empty after registration -- check source data exists")
        if checks["N"] != checks["N_TXN"]:
            raise RuntimeError(
                f"{scoring_view} has {checks['N']} rows for {checks['N_TXN']} transactions -- "
                "the join fans out and would reintroduce duplicate labels"
            )

        return json.dumps(
            {
                "status": "success",
                "step": "feature_engineering",
                "feature_views": ["CUSTOMER_RISK_FEATURES$V1", "TRANSACTION_CONTEXT_FEATURES$V1"],
                "scoring_view": scoring_view,
                "scoring_rows": int(checks["N"]),
            }
        )

    return feature_engineering


def build_train_model_remote(cfg: dict):
    """Build the @remote-decorated training function."""
    pool = cfg["compute_pool"]
    db = cfg["database"]
    schema = cfg["schema"]
    wh = cfg["warehouse"]
    src_db = cfg["source_database"]
    src_schema = cfg["source_schema"]
    scoring_view = SCORING_VIEW_NAME
    feature_cols_cfg = list(FEATURE_COLUMNS)
    stage = f"@{db}.{schema}.PIPELINE_STAGE"
    # Hyperparameters from config (single source of truth -- no second hardcoded copy)
    hp_n_estimators = N_ESTIMATORS
    hp_learning_rate = LEARNING_RATE
    hp_max_depth = MAX_DEPTH
    hp_scale_pos_weight = SCALE_POS_WEIGHT
    # Experiment tracking config (captured by closure)
    exp_enabled = EXPERIMENT_CONFIG.get("enabled", "false") == "true"
    exp_name = EXPERIMENT_CONFIG.get("experiment_name", "FRAUD_DETECTION_TRAINING")
    exp_run_prefix = EXPERIMENT_CONFIG.get("run_name_prefix", "pipeline")

    @remote(
        pool,
        stage_name=stage,
        pip_requirements=["xgboost==3.3.0", "scikit-learn", "snowflake-ml-python"],
    )
    def train_model() -> str:
        import json

        import numpy as np
        import xgboost as xgb
        from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
        from sklearn.model_selection import StratifiedKFold, train_test_split
        from snowflake.snowpark import Session as _Session

        session = _Session.builder.getOrCreate()
        session.sql(f"USE WAREHOUSE {wh}").collect()

        # One row per transaction. The label comes from RAW_TRANSACTIONS joined on
        # TXN_ID (not CUSTOMER_ID), so a given feature vector maps to exactly one label.
        feature_cols = list(feature_cols_cfg)
        select_cols = ",\n                ".join(f"f.{c}" for c in feature_cols)
        df = session.sql(f"""
            SELECT
                f.TXN_ID,
                {select_cols},
                r.IS_FRAUD
            FROM {db}.{schema}.{scoring_view} f
            JOIN {src_db}.{src_schema}.RAW_TRANSACTIONS r ON f.TXN_ID = r.TXN_ID
        """).to_pandas()

        # Guard against the granularity bug returning: identical feature vectors with
        # conflicting labels make the target unlearnable and silently cap precision.
        if df["TXN_ID"].duplicated().any():
            raise RuntimeError("Training set has duplicate TXN_IDs -- the feature join is fanning out")

        X = df[feature_cols].fillna(0)
        y = df["IS_FRAUD"].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        params = {
            "n_estimators": hp_n_estimators,
            "learning_rate": hp_learning_rate,
            "max_depth": hp_max_depth,
            "scale_pos_weight": hp_scale_pos_weight,
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "random_state": 42,
        }

        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        for _, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            fold_model = xgb.XGBClassifier(**params)
            fold_model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx], verbose=False)
            fold_proba = fold_model.predict_proba(X_train.iloc[val_idx])[:, 1]
            cv_scores.append(roc_auc_score(y_train.iloc[val_idx], fold_proba))

        # Final model
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        metrics = {
            "auc_roc": float(roc_auc_score(y_test, y_proba)),
            "pr_auc": float(average_precision_score(y_test, y_proba)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "cv_auc_mean": float(np.mean(cv_scores)),
            "feature_view": scoring_view,
        }

        # Save model artifact to stage
        model.save_model("/tmp/model.ubj")
        session.file.put(
            "/tmp/model.ubj", f"@{db}.{schema}.PIPELINE_STAGE/artifacts/", auto_compress=False, overwrite=True
        )

        # Save sample input
        X_test.head(10).to_json("/tmp/sample_input.json", orient="records")
        session.file.put(
            "/tmp/sample_input.json", f"@{db}.{schema}.PIPELINE_STAGE/artifacts/", auto_compress=False, overwrite=True
        )

        # Experiment tracking
        if exp_enabled:
            from datetime import datetime

            from snowflake.ml.experiment import ExperimentTracking

            exp = ExperimentTracking(session=session, database_name=db, schema_name=schema)
            exp.set_experiment(exp_name)
            run_name = f"{exp_run_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with exp.start_run(run_name):
                exp.log_params(params)
                exp.log_params({"feature_view": scoring_view, "dataset_rows": str(len(df)), "test_size": "0.2"})
                # log_metrics only accepts numeric values — filter out strings
                numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                exp.log_metrics(numeric_metrics)

        # Write metrics to results table
        result = {"status": "success", "step": "training", "metrics": metrics}
        result_json = json.dumps(result).replace("'", "''")
        session.sql(f"""
            INSERT INTO {db}.{schema}.PIPELINE_RESULTS (STEP, STATUS, RESULT, CREATED_AT)
            SELECT 'training', 'SUCCESS', PARSE_JSON('{result_json}'), CURRENT_TIMESTAMP()
        """).collect()

        return json.dumps(result)

    return train_model


def build_evaluate_remote(cfg: dict):
    """Build the @remote-decorated evaluation function."""
    pool = cfg["compute_pool"]
    db = cfg["database"]
    schema = cfg["schema"]
    wh = cfg["warehouse"]
    stage = f"@{db}.{schema}.PIPELINE_STAGE"

    @remote(
        pool,
        stage_name=stage,
        pip_requirements=["snowflake-ml-python"],
    )
    def evaluate_model() -> str:
        import json

        from snowflake.snowpark import Session as _Session

        session = _Session.builder.getOrCreate()
        session.sql(f"USE WAREHOUSE {wh}").collect()

        rows = session.sql(f"""
            SELECT RESULT FROM {db}.{schema}.PIPELINE_RESULTS
            WHERE STEP = 'training' AND STATUS = 'SUCCESS'
            ORDER BY CREATED_AT DESC LIMIT 1
        """).collect()

        if not rows:
            return json.dumps({"status": "error", "message": "No training results found"})

        training_result = json.loads(rows[0]["RESULT"])
        metrics = training_result.get("metrics", {})

        result = {"status": "success", "step": "evaluation", "metrics": metrics, "pipeline_status": "READY_FOR_REVIEW"}
        result_json = json.dumps(result).replace("'", "''")
        session.sql(f"""
            INSERT INTO {db}.{schema}.PIPELINE_RESULTS (STEP, STATUS, RESULT, CREATED_AT)
            SELECT 'evaluation', 'SUCCESS', PARSE_JSON('{result_json}'), CURRENT_TIMESTAMP()
        """).collect()

        return json.dumps(result)

    return evaluate_model


# ─── Warehouse-based alternatives (StoredProcedureCall) ──────────────────────


def feature_eng_warehouse(session: Session) -> str:
    """Feature engineering on warehouse — registers Feature Views + scoring view."""
    from features.feature_views import register_feature_views, register_scoring_view

    register_feature_views(session=session)
    register_scoring_view(session=session)
    return "feature_engineering_complete"


def train_model_warehouse(session: Session) -> str:
    """Not supported: warehouse Python UDFs cannot install xgboost/sklearn builds.

    Kept so the symbol exists, but deploy_dag() rejects training_compute="warehouse"
    up front rather than deploying a task that always fails at runtime.
    """
    raise NotImplementedError("Training requires SPCS compute pool for custom packages (xgboost, sklearn)")


def evaluate_warehouse(session: Session) -> str:
    """Evaluation on warehouse — reads metrics from results table."""
    rows = session.sql("""
        SELECT RESULT FROM PIPELINE_RESULTS
        WHERE STEP = 'training' AND STATUS = 'SUCCESS'
        ORDER BY CREATED_AT DESC LIMIT 1
    """).collect()
    if rows:
        return "evaluation_complete"
    return "evaluation_error_no_training_results"


# ─── DAG Deployment ──────────────────────────────────────────────────────────


def deploy_dag(env: str):
    """Deploy the ML Training Pipeline DAG using Python SDK."""
    cfg = get_env_config(env)
    db = cfg["database"]
    schema = cfg["schema"]
    wh = cfg["warehouse"]

    fe_compute = PIPELINE_CONFIG.get("feature_engineering_compute", "warehouse")
    train_compute = PIPELINE_CONFIG.get("training_compute", "spcs")
    eval_compute = PIPELINE_CONFIG.get("evaluation_compute", "spcs")

    # Fail fast rather than deploying a TRAIN_MODEL task that always errors at runtime.
    if train_compute != "spcs":
        raise ValueError(
            f'training_compute="{train_compute}" is not supported. Training needs xgboost and '
            "scikit-learn, which warehouse Python UDFs cannot provide. Set "
            'PIPELINE_CONFIG["training_compute"] = "spcs" in source/config.py.'
        )

    # Task timeout: Snowflake defaults to 1h, which is shorter than this pipeline
    # may need on a cold compute pool (image pull + package install).
    task_timeout_ms = int(PIPELINE_CONFIG.get("task_timeout_ms", "7200000"))

    session = create_snowpark_session()
    session.sql(f"USE WAREHOUSE {wh}").collect()

    print(f"Deploying DAG: {db}.{schema}.{DAG_NAME}")
    print(f"  Feature eng: {fe_compute} | Training: {train_compute} | Evaluation: {eval_compute}")
    print(f"  Task timeout: {task_timeout_ms} ms ({task_timeout_ms / 3_600_000:.1f}h)")

    # Create infrastructure
    session.sql(f"CREATE STAGE IF NOT EXISTS {db}.{schema}.PIPELINE_STAGE").collect()
    session.sql(f"""
        CREATE TABLE IF NOT EXISTS {db}.{schema}.PIPELINE_RESULTS (
            STEP VARCHAR, STATUS VARCHAR, RESULT VARIANT, CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        )
    """).collect()

    # Build the DAG
    root = Root(session)
    stage_location = f"@{db}.{schema}.PIPELINE_STAGE"

    # Add schedule for periodic retraining (if enabled)
    dag_schedule = None
    if RETRAIN_CONFIG.get("enabled", "false") == "true":
        schedule_str = RETRAIN_CONFIG.get("schedule", "")
        if schedule_str:
            from snowflake.core.task import Cron

            # Parse "USING CRON <expr> <tz>" format
            parts = schedule_str.replace("USING CRON ", "").strip()
            # Split: last token is timezone, rest is cron expression
            tokens = parts.split()
            tz = tokens[-1]
            cron_expr = " ".join(tokens[:-1])
            dag_schedule = Cron(cron_expr, tz)
            print(f"  Schedule: {cron_expr} ({tz})")

    with DAG(
        DAG_NAME,
        warehouse=wh,
        schedule=dag_schedule,
        user_task_timeout_ms=task_timeout_ms,
    ) as dag:
        # Feature Engineering task
        if fe_compute == "spcs":
            fe_func = build_feature_eng_remote(cfg)
            fe_task = DAGTask("FEATURE_ENG", definition=fe_func, user_task_timeout_ms=task_timeout_ms)
        else:
            fe_task = DAGTask(
                "FEATURE_ENG",
                StoredProcedureCall(
                    feature_eng_warehouse,
                    stage_location=stage_location,
                    packages=["snowflake-ml-python", "snowflake-snowpark-python"],
                ),
                warehouse=wh,
                user_task_timeout_ms=task_timeout_ms,
            )

        # Training task (spcs only -- validated above)
        train_func = build_train_model_remote(cfg)
        train_task = DAGTask("TRAIN_MODEL", definition=train_func, user_task_timeout_ms=task_timeout_ms)

        # Evaluation task
        if eval_compute == "spcs":
            eval_func = build_evaluate_remote(cfg)
            eval_task = DAGTask("EVALUATE", definition=eval_func, user_task_timeout_ms=task_timeout_ms)
        else:
            eval_task = DAGTask(
                "EVALUATE",
                StoredProcedureCall(
                    evaluate_warehouse, stage_location=stage_location, packages=["snowflake-snowpark-python"]
                ),
                warehouse=wh,
                user_task_timeout_ms=task_timeout_ms,
            )

        # Define dependencies
        fe_task >> train_task >> eval_task

    # Deploy
    schema_ref = root.databases[db].schemas[schema]
    dag_op = DAGOperation(schema_ref)
    dag_op.deploy(dag, mode=CreateMode.or_replace)
    print(f"  DAG deployed: {DAG_NAME}")

    # Only resume root task if explicitly requested (scheduled-retrain workflow sets RESUME_SCHEDULE=true).
    # Do NOT auto-resume — otherwise cron fires during CI runs and interferes with wait_for_task.
    if os.getenv("RESUME_SCHEDULE", "false").lower() == "true" and dag_schedule:
        session.sql(f"ALTER TASK {db}.{schema}.{DAG_NAME} RESUME").collect()
        print("  Root task resumed (scheduled retraining active)")

    session.close()
    print("Deploy complete.")


def execute_dag(env: str):
    """Trigger the DAG execution."""
    cfg = get_env_config(env)
    db = cfg["database"]
    schema = cfg["schema"]
    wh = cfg["warehouse"]

    session = create_snowpark_session()
    session.sql(f"USE WAREHOUSE {wh}").collect()

    # Clear previous results
    session.sql(f"DELETE FROM {db}.{schema}.PIPELINE_RESULTS").collect()

    # Execute the root task
    root = Root(session)
    task_res = root.databases[db].schemas[schema].tasks[DAG_NAME]
    task_res.execute()
    print(f"Executed: {db}.{schema}.{DAG_NAME}")

    session.close()


def show_status(env: str):
    """Show recent task execution history."""
    cfg = get_env_config(env)
    db = cfg["database"]
    wh = cfg["warehouse"]

    session = create_snowpark_session()
    session.sql(f"USE WAREHOUSE {wh}").collect()

    print(f"Task history for {db} (last 24h):\n")
    rows = session.sql(f"""
        SELECT NAME, STATE, SCHEDULED_TIME, COMPLETED_TIME, RETURN_VALUE, ERROR_MESSAGE
        FROM TABLE(INFORMATION_SCHEMA.TASK_HISTORY(
            SCHEDULED_TIME_RANGE_START => DATEADD('hour', -24, CURRENT_TIMESTAMP()),
            RESULT_LIMIT => 20
        ))
        WHERE DATABASE_NAME = '{db}'
        ORDER BY SCHEDULED_TIME DESC
    """).collect()

    if not rows:
        print("  No task runs found in the last 24 hours.")
    else:
        for row in rows:
            ts = str(row["SCHEDULED_TIME"])[:19]
            state = row["STATE"]
            name = row["NAME"]
            ret = row["RETURN_VALUE"] or ""
            err = row["ERROR_MESSAGE"] or ""
            print(f"  [{ts}] {name}: {state} {ret[:80]} {err[:80]}")

    session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Pipeline Task DAG (Python SDK)")
    parser.add_argument("--deploy", action="store_true", help="Deploy the Task DAG")
    parser.add_argument("--execute", action="store_true", help="Execute the Task DAG")
    parser.add_argument("--status", action="store_true", help="Show recent task history")
    parser.add_argument("--env", default=os.getenv("ML_ENV", "dev"), choices=["dev", "stage"])
    args = parser.parse_args()

    if args.deploy:
        deploy_dag(args.env)
    elif args.execute:
        execute_dag(args.env)
    elif args.status:
        show_status(args.env)
    else:
        parser.print_help()
