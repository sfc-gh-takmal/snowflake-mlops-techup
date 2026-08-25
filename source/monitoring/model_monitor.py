"""Model Monitor: create and manage ML Observability for the fraud detector.

Uses Snowflake's native ModelMonitor (CREATE MODEL MONITOR) to track
prediction drift and feature distribution shifts over time.

The monitor reads from the BATCH_PREDICTIONS table (scored by batch inference)
and checks for distribution shifts at a configurable interval.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MODEL_NAME, MONITOR_CONFIG


def _row_dict(row) -> dict:
    """Snowpark Row is tuple-derived and has no dict-style .get(); convert first."""
    return row.as_dict() if row is not None else {}


def _describe(session, db: str, schema: str, monitor_name: str) -> dict:
    """DESC MODEL MONITOR as a plain dict, or {} if the monitor is absent."""
    try:
        rows = session.sql(f"DESC MODEL MONITOR {db}.{schema}.{monitor_name}").collect()
    except Exception:
        return {}
    return _row_dict(rows[0]) if rows else {}


def _monitored_version(desc: dict) -> str:
    """Extract the monitored version name from DESC output.

    DESC MODEL MONITOR returns the model details as a JSON string in the
    'model' column (fields: model_name, version_name, function_name, ...).
    There is no top-level 'model_version_name' column.
    """
    raw = desc.get("model") or ""
    if not raw:
        return ""
    try:
        return json.loads(raw).get("version_name", "")
    except (ValueError, TypeError):
        return ""


def get_env_config(env: str) -> dict:
    """Get environment-specific database/warehouse for monitor."""
    envs = {
        "dev": {"database": "SNOW_MLOPS_DEV", "warehouse": "SNOW_MLOPS_DEV_WH"},
        "stage": {"database": "SNOW_MLOPS_STAGE", "warehouse": "SNOW_MLOPS_STAGE_WH"},
        "prod": {"database": "SNOW_MLOPS_PROD", "warehouse": "SNOW_MLOPS_PROD_WH"},
    }
    return envs.get(env, envs["dev"])


def create_monitor(session, env: str = "prod") -> str:
    """Create the model monitor if it doesn't exist.

    Returns the monitor name.
    """
    cfg = get_env_config(env)
    db = os.getenv("SNOWFLAKE_DATABASE", cfg["database"])
    schema = os.getenv("SNOWFLAKE_SCHEMA", "ML")
    wh = cfg["warehouse"]

    monitor_name = MONITOR_CONFIG["monitor_name"]
    source_table = f"{db}.{schema}.{MONITOR_CONFIG['source_table']}"
    ts_col = MONITOR_CONFIG["timestamp_column"]
    pred_cols = MONITOR_CONFIG["prediction_columns"]
    refresh = MONITOR_CONFIG["refresh_interval"]
    agg_window = MONITOR_CONFIG["aggregation_window"]
    function_name = MONITOR_CONFIG["function_name"]

    # Get current default model version
    models = session.sql(f"SHOW MODELS LIKE '{MODEL_NAME}' IN {db}.{schema}").collect()
    if not models:
        raise RuntimeError(f"Model {MODEL_NAME} not found in {db}.{schema}")
    version = models[0]["default_version_name"]

    pred_cols_sql = ", ".join(f"'{c}'" for c in pred_cols)
    # ID_COLUMNS must uniquely identify a row in SOURCE. A column may be used only
    # once across all monitor parameters, so declaring TXN_ID here also stops it
    # being treated as a categorical feature.
    id_cols_sql = ", ".join(f"'{c}'" for c in MONITOR_CONFIG.get("id_columns", ["TXN_ID"]))

    # Check if monitor already exists
    existing = session.sql(f"SHOW MODEL MONITORS LIKE '{monitor_name}' IN SCHEMA {db}.{schema}").collect()

    if existing:
        print(f"  Monitor {monitor_name} already exists. Verifying state...")
        # Check if it's tracking the right version
        current_version = _monitored_version(_describe(session, db, schema, monitor_name))
        if current_version != version:
            print(f"  Updating monitor to track version {version} (was {current_version})...")
            # Drop and recreate (ALTER not supported for version change)
            session.sql(f"DROP MODEL MONITOR IF EXISTS {db}.{schema}.{monitor_name}").collect()
        else:
            print(f"  Monitor is tracking {MODEL_NAME}/{version}. No changes needed.")
            _ensure_active(session, db, schema, monitor_name)
            return monitor_name

    # Create the monitor
    print(f"  Creating monitor: {monitor_name}")
    print(f"    Model: {MODEL_NAME}/{version}")
    print(f"    Source: {source_table}")
    print(f"    Refresh: {refresh}, Aggregation: {agg_window}")

    session.sql(f"""
        CREATE MODEL MONITOR {db}.{schema}.{monitor_name} WITH
            MODEL = {db}.{schema}.{MODEL_NAME}
            VERSION = '{version}'
            FUNCTION = '{function_name}'
            SOURCE = {source_table}
            ID_COLUMNS = ({id_cols_sql})
            WAREHOUSE = {wh}
            REFRESH_INTERVAL = '{refresh}'
            AGGREGATION_WINDOW = '{agg_window}'
            TIMESTAMP_COLUMN = {ts_col}
            PREDICTION_SCORE_COLUMNS = ({pred_cols_sql})
    """).collect()

    print("  Monitor created successfully.")
    _ensure_active(session, db, schema, monitor_name)
    return monitor_name


def _ensure_active(session, db: str, schema: str, monitor_name: str):
    """Ensure the monitor is in ACTIVE state."""
    try:
        desc = _describe(session, db, schema, monitor_name)
        if desc:
            state = desc.get("monitor_state", "UNKNOWN")
            if state in ("SUSPENDED", "PARTIALLY_SUSPENDED"):
                print(f"  Monitor is {state}; resuming...")
                if desc.get("aggregation_last_error"):
                    print(f"  Last aggregation error: {desc['aggregation_last_error']}")
                session.sql(f"ALTER MODEL MONITOR {db}.{schema}.{monitor_name} RESUME").collect()
            elif state == "ACTIVE":
                print("  Monitor state: ACTIVE")
            else:
                print(f"  Monitor state: {state}")
    except Exception as e:
        print(f"  Warning: could not check monitor state: {e}")


def get_monitor_status(session, env: str = "prod") -> dict:
    """Get the current monitor status and latest metrics."""
    cfg = get_env_config(env)
    db = os.getenv("SNOWFLAKE_DATABASE", cfg["database"])
    schema = os.getenv("SNOWFLAKE_SCHEMA", "ML")
    monitor_name = MONITOR_CONFIG["monitor_name"]

    desc = _describe(session, db, schema, monitor_name)
    if not desc:
        return {"exists": False}
    return {
        "exists": True,
        "state": desc.get("monitor_state", "UNKNOWN"),
        "model_version": _monitored_version(desc),
        "refresh_interval": desc.get("refresh_interval", ""),
        "aggregation_status": desc.get("aggregation_status", ""),
        "aggregation_last_error": desc.get("aggregation_last_error", ""),
    }


def suspend_monitor(session, env: str = "prod"):
    """Suspend the monitor (e.g., during maintenance)."""
    cfg = get_env_config(env)
    db = os.getenv("SNOWFLAKE_DATABASE", cfg["database"])
    schema = os.getenv("SNOWFLAKE_SCHEMA", "ML")
    monitor_name = MONITOR_CONFIG["monitor_name"]
    session.sql(f"ALTER MODEL MONITOR {db}.{schema}.{monitor_name} SUSPEND").collect()
    print(f"  Monitor {monitor_name} suspended.")


def resume_monitor(session, env: str = "prod"):
    """Resume a suspended monitor."""
    cfg = get_env_config(env)
    db = os.getenv("SNOWFLAKE_DATABASE", cfg["database"])
    schema = os.getenv("SNOWFLAKE_SCHEMA", "ML")
    monitor_name = MONITOR_CONFIG["monitor_name"]
    session.sql(f"ALTER MODEL MONITOR {db}.{schema}.{monitor_name} RESUME").collect()
    print(f"  Monitor {monitor_name} resumed.")
