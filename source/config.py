"""Centralized configuration for the Snowflake MLOps demo."""

# Target environment (where the pipeline WRITES: features, models, experiments, services)
DATABASE = "SNOW_MLOPS_DEV"
SCHEMA = "ML"
WAREHOUSE = "SNOW_MLOPS_DEV_WH"
COMPUTE_POOL = "SNOW_MLOPS_DEV_POOL"

# Source environment (where raw data LIVES -- always PROD)
SOURCE_DATABASE = "SNOW_MLOPS_PROD"
SOURCE_SCHEMA = "ML"

FULLY_QUALIFIED_SCHEMA = f"{DATABASE}.{SCHEMA}"

# Stages (in target environment)
ML_ARTIFACTS_STAGE = f"@{DATABASE}.{SCHEMA}.ML_ARTIFACTS"
DAG_STAGE = f"@{DATABASE}.{SCHEMA}.DAG_STAGE"
JOB_STAGE = f"@{DATABASE}.{SCHEMA}.JOB_STAGE"

# Source tables (read-only, from PROD)
RAW_TRANSACTIONS_TABLE = f"{SOURCE_DATABASE}.{SOURCE_SCHEMA}.RAW_TRANSACTIONS"
CUSTOMER_PROFILES_TABLE = f"{SOURCE_DATABASE}.{SOURCE_SCHEMA}.CUSTOMER_PROFILES"
MERCHANT_DATA_TABLE = f"{SOURCE_DATABASE}.{SOURCE_SCHEMA}.MERCHANT_DATA"

# Model
MODEL_NAME = "MLOPS_FRAUD_DETECTOR"
SERVICE_NAME = "MLOPS_FRAUD_DETECTOR_SERVICE"

# Feature Store
FEATURE_STORE_SCHEMA = SCHEMA
FEATURE_VIEW_NAME = "CUSTOMER_RISK_FEATURES"
FEATURE_VIEW_VERSION = "V1"  # Bump when feature SQL changes
TXN_FEATURE_VIEW_NAME = "TRANSACTION_CONTEXT_FEATURES"
TXN_FEATURE_VIEW_VERSION = "V1"

# Scoring view: joins the per-transaction and per-customer feature views into one
# row per transaction. Training, batch inference and the real-time service all
# read the same columns from here, so the feature contract cannot drift between them.
SCORING_VIEW_NAME = "FRAUD_SCORING_FEATURES"

# The model's feature contract -- SINGLE SOURCE OF TRUTH.
# Previously these 12 column names were duplicated across the training job, the
# SPCS health check and the endpoint tests, and had to be kept in sync by hand.
#
# Prediction is per transaction, so transaction-level context and customer history
# are combined. HISTORICAL_FRAUD_COUNT and HISTORICAL_FRAUD_RATE are deliberately
# EXCLUDED: they are aggregates of IS_FRAUD over the same transactions being
# scored, so including them leaks the label into the features.
TRANSACTION_FEATURE_COLUMNS = [
    "AMOUNT",
    "AMOUNT_TO_AVG_RATIO",
    "IS_HIGH_RISK_MERCHANT",
    "MERCHANT_RISK_SCORE",
    "HOUR_OF_DAY",
    "IS_WEEKEND",
    "IS_LATE_NIGHT",
]
CUSTOMER_FEATURE_COLUMNS = [
    "TOTAL_TXN_COUNT",
    "AVG_TXN_AMOUNT",
    "MAX_TXN_AMOUNT",
    "STDDEV_TXN_AMOUNT",
    "UNIQUE_MERCHANTS",
    "ACTIVE_DAYS",
    "LATE_NIGHT_TXN_RATIO",
    "CREDIT_SCORE",
    "ACCOUNT_AGE_DAYS",
    "ANNUAL_INCOME",
]
FEATURE_COLUMNS = TRANSACTION_FEATURE_COLUMNS + CUSTOMER_FEATURE_COLUMNS
LABEL_COLUMN = "IS_FRAUD"
ID_COLUMN = "TXN_ID"

# Quality gate thresholds (model must meet ALL to promote to PROD).
# SINGLE SOURCE OF TRUTH -- PIPELINE_CONFIG below derives from these, so there is
# no second set of values that can silently disagree with what CI enforces.
#
# Calibrated against observed per-transaction model performance on the synthetic
# dataset (AUC 0.895, precision 0.196, recall 0.665, CV AUC 0.887), leaving
# headroom so normal run-to-run variance passes but a real regression fails.
# Base fraud rate is 3%, so precision 0.12 is a 4x lift over guessing.
MIN_AUC_ROC = 0.80
MIN_PRECISION = 0.12
MIN_RECALL = 0.50

# Training hyperparameters (single source of truth; the @remote training
# function reads these rather than hardcoding its own copy).
N_ESTIMATORS = 200
LEARNING_RATE = 0.1
MAX_DEPTH = 6
SCALE_POS_WEIGHT = 33

# Pipeline defaults
PIPELINE_CONFIG = {
    "database": DATABASE,
    "schema": SCHEMA,
    "source_database": SOURCE_DATABASE,
    "source_schema": SOURCE_SCHEMA,
    "warehouse": WAREHOUSE,
    "compute_pool": COMPUTE_POOL,
    # Compute mode for each pipeline step: "warehouse" or "spcs"
    "feature_engineering_compute": "spcs",
    "training_compute": "spcs",
    "evaluation_compute": "spcs",
    # Training hyperparameters (derived -- do not edit here, edit the constants)
    "n_estimators": str(N_ESTIMATORS),
    "learning_rate": str(LEARNING_RATE),
    "max_depth": str(MAX_DEPTH),
    "scale_pos_weight": str(SCALE_POS_WEIGHT),
    # Evaluation thresholds (derived -- do not edit here, edit the constants)
    "min_auc_roc": str(MIN_AUC_ROC),
    "min_precision": str(MIN_PRECISION),
    "min_recall": str(MIN_RECALL),
    # Deployment
    "model_name": MODEL_NAME,
    "service_name": SERVICE_NAME,
    "max_instances": "2",
    # Deployment toggles
    "deploy_batch_inference": "true",
    "deploy_realtime_service": "true",
    "enable_model_monitor": "true",
    # Task configuration
    "task_timeout_ms": "7200000",  # 2 hours (max: 86400000 = 24h)
    # Internal stage for pipeline code
    "pipeline_stage": f"@{DATABASE}.{SCHEMA}.PIPELINE_STAGE",
}

# Model Monitor configuration
MONITOR_CONFIG = {
    "monitor_name": "FRAUD_DETECTOR_MONITOR",
    "function_name": "predict_proba",
    "source_table": "BATCH_PREDICTIONS",
    "timestamp_column": "PREDICTION_TS",
    "id_columns": [ID_COLUMN],
    "prediction_columns": ["output_feature_1"],  # P(fraud)
    "refresh_interval": "1 day",
    "aggregation_window": "7 days",
}

# Experiment Tracking configuration
EXPERIMENT_CONFIG = {
    "enabled": "true",
    "experiment_name": "FRAUD_DETECTION_TRAINING",
    "run_name_prefix": "pipeline",  # run name = prefix + timestamp
}

# Feature View refresh configuration
FEATURE_VIEW_CONFIG = {
    "customer_features_refresh": "1 hour",  # TARGET_LAG for CUSTOMER_RISK_FEATURES
    "transaction_features_refresh": "1 hour",  # TARGET_LAG for TRANSACTION_CONTEXT_FEATURES
}

# Scheduled Retraining configuration
RETRAIN_CONFIG = {
    "enabled": "true",
    "schedule": "USING CRON 0 6 * * MON America/Los_Angeles",  # Every Monday 6AM PT
    "stage_only": "true",  # Train + register in STAGE only; human promotes to PROD
    "notify_github_issue": "true",  # Create GitHub issue when candidate is ready
}
