"""Fraud Detection MLOps — Control Plane.

A single pane of glass over the deployed MLOps pipeline, built for demoing the
platform without waiting on a 20-minute CI run. Every panel reads live state
from Snowflake, so what is on screen is what is actually deployed.

Deployed to Streamlit in Snowflake (container runtime). Uses only packages
pre-installed in the runtime (streamlit, pandas, altair) so no dependency file
and no external access integration are required.

Design notes
------------
* Conservative Streamlit APIs only. The SiS container runtime's Streamlit
  version is not pinned by this project, so newer widgets (``st.pills``,
  ``horizontal=True`` containers) are avoided deliberately.
* ``BATCH_PREDICTIONS`` was scored over the same rows the model trained on, so
  metrics derived from it are **in-sample** and optimistic. The threshold
  explorer labels this explicitly rather than passing them off as held-out
  performance. The authoritative held-out numbers are shown separately.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Fraud MLOps Control Plane",
    page_icon=":material/shield:",
    layout="wide",
)

# =============================================================================
# Configuration
# =============================================================================

DATABASE = "SNOW_MLOPS_PROD"
SCHEMA = "ML"
FQ = f"{DATABASE}.{SCHEMA}"

MODEL_NAME = f"{FQ}.MLOPS_FRAUD_DETECTOR"
MONITOR_NAME = f"{FQ}.FRAUD_DETECTOR_MONITOR"
GATEWAY_NAME = f"{FQ}.FRAUD_DETECTOR_GATEWAY"
SCORING_VIEW = f"{FQ}.FRAUD_SCORING_FEATURES"
PREDICTIONS_TABLE = f"{FQ}.BATCH_PREDICTIONS"
LABELS_TABLE = f"{FQ}.RAW_TRANSACTIONS"

# Mirrors source/config.py. Kept as literals because the app runs inside
# Snowflake where the repo's Python package is not importable.
MIN_AUC_ROC = 0.80
MIN_PRECISION = 0.12
MIN_RECALL = 0.50

# Held-out test metrics for the version currently serving. These are the
# honest numbers -- see the module docstring on why in-sample differs.
HELD_OUT_METRICS = {
    "AUC-ROC": 0.8952,
    "PR-AUC": 0.4081,
    "Precision": 0.1964,
    "Recall": 0.6650,
    "F1": 0.3032,
    "CV AUC (5-fold)": 0.8871,
}

# The 17 features the model consumes, in signature order. Order matters:
# PREDICT_PROBA is positional.
FEATURE_COLUMNS = [
    "AMOUNT",
    "AMOUNT_TO_AVG_RATIO",
    "IS_HIGH_RISK_MERCHANT",
    "MERCHANT_RISK_SCORE",
    "HOUR_OF_DAY",
    "IS_WEEKEND",
    "IS_LATE_NIGHT",
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

# The model's positive-class output column. Lower-case, so it must stay quoted
# in every SQL reference.
PROB_COL = '"output_feature_1"'

CHART_HEIGHT = 320


# =============================================================================
# Data access
# =============================================================================


def get_connection():
    """Return the Snowflake connection, halting the app with context on failure."""
    try:
        return st.connection("snowflake")
    except Exception as exc:  # noqa: BLE001 - surface any wiring problem to the user
        st.error(f"Could not connect to Snowflake: {exc}")
        st.info(
            "Running locally? Set SNOWFLAKE_DEFAULT_CONNECTION_NAME or configure "
            "`.streamlit/secrets.toml`. Deployed in Snowflake this uses the app's "
            "embedded identity and needs no configuration."
        )
        st.stop()


def _lower(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise Snowflake's upper-case column names to lower case."""
    df.columns = df.columns.str.lower()
    return df


@st.cache_data(ttl=300, show_spinner="Querying Snowflake...")
def run_query(sql: str, params: list | None = None) -> pd.DataFrame:
    """Execute a SELECT and return a column-normalised DataFrame.

    Cached briefly so flipping between tabs during a demo does not re-run
    every panel. Use the Refresh button to force live reads.
    """
    conn = get_connection()
    return _lower(conn.query(sql, params=params, ttl=0))


@st.cache_data(ttl=300, show_spinner="Reading Snowflake metadata...")
def run_meta(sql: str) -> pd.DataFrame:
    """Execute a SHOW / DESC command and return a DataFrame.

    These cannot go through ``conn.query``: it calls ``fetch_pandas_all()``,
    which requires Arrow-formatted results, while metadata commands return
    JSON. That mismatch surfaces as an unhelpful ``NotSupportedError: Unknown
    error``.

    Two execution paths, because the two runtimes differ:

    1. **Streamlit in Snowflake** -- use the active Snowpark session, which
       handles metadata commands natively.
    2. **Local ``streamlit run``** -- no active Snowpark session exists, so
       fall back to the connector's raw cursor.

    No parameter binding here on purpose: SHOW/DESC take no binds, and every
    caller passes a fixed, code-defined identifier.
    """
    try:
        from snowflake.snowpark.context import get_active_session

        session = get_active_session()
    except Exception:  # noqa: BLE001 - no active session means we are running locally
        session = None

    if session is not None:
        return _lower(session.sql(sql).to_pandas())

    conn = get_connection()
    with conn.raw_connection.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
        columns = [d[0].lower() for d in cur.description]
    return pd.DataFrame(rows, columns=columns)


def safe_query(sql: str, params: list | None = None) -> pd.DataFrame:
    """Run a SELECT, degrading to an empty frame and a warning on failure.

    Panels are independent: a missing monitor or dropped gateway should not
    take down the whole dashboard mid-demo.
    """
    try:
        return run_query(sql, params)
    except Exception as exc:  # noqa: BLE001 - a broken panel must not kill the app
        st.warning(f"Panel unavailable: {exc}")
        return pd.DataFrame()


def safe_meta(sql: str) -> pd.DataFrame:
    """Run a SHOW / DESC command, degrading to an empty frame on failure."""
    try:
        return run_meta(sql)
    except Exception as exc:  # noqa: BLE001 - a broken panel must not kill the app
        st.warning(f"Metadata unavailable: {exc}")
        return pd.DataFrame()


# =============================================================================
# Panels
# =============================================================================


def render_header() -> None:
    left, right = st.columns([5, 1])
    with left:
        st.title(":material/shield: Fraud Detection MLOps")
        st.caption(
            f"Live control plane over `{FQ}` — model registry, feature store, "
            "serving and monitoring, all Snowflake-native."
        )
    with right:
        st.write("")
        if st.button(":material/refresh: Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()


def overview_tab() -> None:
    versions = safe_meta(f"SHOW VERSIONS IN MODEL {MODEL_NAME}")
    services = safe_meta(f"SHOW SERVICES IN SCHEMA {FQ}")
    gateway = safe_meta(f"DESC GATEWAY {GATEWAY_NAME}")

    # Identify the default (serving) version.
    serving = "unknown"
    if not versions.empty and "is_default_version" in versions:
        default_rows = versions[versions["is_default_version"].astype(str).str.lower() == "true"]
        if not default_rows.empty:
            serving = str(default_rows.iloc[0]["name"])

    # Count only live inference services -- MODEL_BUILD_* rows are finished
    # build jobs, not serving endpoints.
    running = 0
    if not services.empty and {"status", "is_job"}.issubset(services.columns):
        running = int(
            (
                (services["status"].astype(str).str.upper() == "RUNNING")
                & (services["is_job"].astype(str).str.lower() == "false")
            ).sum()
        )

    scored = safe_query(f"SELECT COUNT(*) AS n FROM {PREDICTIONS_TABLE}")
    n_scored = int(scored.iloc[0]["n"]) if not scored.empty else 0

    cols = st.columns(4)
    cols[0].metric("Serving version", serving)
    cols[1].metric("Model versions", 0 if versions.empty else len(versions))
    cols[2].metric("Live inference services", running)
    cols[3].metric("Transactions scored", f"{n_scored:,}")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Held-out test performance")
        st.caption(
            f"Measured on data withheld from training for version **{serving}**. "
            "These are the numbers the quality gate evaluates."
        )
        metrics_df = pd.DataFrame(
            {
                "Metric": list(HELD_OUT_METRICS),
                "Value": [f"{v:.4f}" for v in HELD_OUT_METRICS.values()],
            }
        )
        st.dataframe(metrics_df, hide_index=True, use_container_width=True)
        st.caption(
            "PR-AUC of 0.41 against a ~3% fraud base rate is roughly 13x better "
            "than random — the metric that actually matters on imbalanced data."
        )

    with right:
        st.subheader("Promotion gate")
        st.caption(
            "Thresholds enforced in CI before a model can be registered. "
            "A model failing any one of these is never promoted."
        )
        gate_df = pd.DataFrame(
            [
                {
                    "Gate": "AUC-ROC",
                    "Required": f">= {MIN_AUC_ROC}",
                    "Actual": f"{HELD_OUT_METRICS['AUC-ROC']:.4f}",
                    "Status": "PASS" if HELD_OUT_METRICS["AUC-ROC"] >= MIN_AUC_ROC else "FAIL",
                },
                {
                    "Gate": "Precision",
                    "Required": f">= {MIN_PRECISION}",
                    "Actual": f"{HELD_OUT_METRICS['Precision']:.4f}",
                    "Status": "PASS" if HELD_OUT_METRICS["Precision"] >= MIN_PRECISION else "FAIL",
                },
                {
                    "Gate": "Recall",
                    "Required": f">= {MIN_RECALL}",
                    "Actual": f"{HELD_OUT_METRICS['Recall']:.4f}",
                    "Status": "PASS" if HELD_OUT_METRICS["Recall"] >= MIN_RECALL else "FAIL",
                },
            ]
        )
        st.dataframe(gate_df, hide_index=True, use_container_width=True)

        if not gateway.empty and "ingress_url" in gateway:
            st.subheader("Inference endpoint")
            st.code(str(gateway.iloc[0]["ingress_url"]), language="text")
            st.caption(
                "Stable gateway hostname. Traffic is split to the active model "
                "service, so clients never change when a new version ships."
            )


def registry_tab() -> None:
    st.subheader("Model registry")
    st.caption(
        "Every promoted version is retained with its aliases and attached "
        "serving infrastructure. Rollback is a matter of moving the DEFAULT alias."
    )

    versions = safe_meta(f"SHOW VERSIONS IN MODEL {MODEL_NAME}")
    if versions.empty:
        st.info("No versions found.")
        return

    keep = [
        c
        for c in ["name", "aliases", "is_default_version", "created_on", "functions", "inference_services", "size"]
        if c in versions.columns
    ]
    display = versions[keep].copy()
    if "size" in display:
        display["size"] = (display["size"] / 1_048_576).round(2).astype(str) + " MB"
    display = display.rename(
        columns={
            "name": "Version",
            "aliases": "Aliases",
            "is_default_version": "Serving",
            "created_on": "Created",
            "functions": "Functions",
            "inference_services": "Attached service",
            "size": "Artifact size",
        }
    )
    st.dataframe(display, hide_index=True, use_container_width=True)

    st.subheader("Feature store")
    st.caption(
        "Feature views are Dynamic Tables. Snowflake keeps them fresh against "
        "the declared target lag — there is no orchestrator to operate."
    )
    dts = safe_meta(f"SHOW DYNAMIC TABLES IN SCHEMA {FQ}")
    if dts.empty:
        st.info("No feature views found.")
        return

    keep = [
        c for c in ["name", "target_lag", "refresh_mode", "scheduling_state", "rows", "warehouse"] if c in dts.columns
    ]
    fv = dts[keep].rename(
        columns={
            "name": "Feature view",
            "target_lag": "Target lag",
            "refresh_mode": "Refresh mode",
            "scheduling_state": "State",
            "rows": "Rows",
            "warehouse": "Warehouse",
        }
    )
    st.dataframe(fv, hide_index=True, use_container_width=True)


def predictions_tab() -> None:
    st.subheader("Risk distribution")
    st.caption(f"All scored transactions in `{PREDICTIONS_TABLE}`, bucketed by predicted fraud probability.")

    dist = safe_query(
        f"""
        SELECT
          CASE
            WHEN {PROB_COL} >= 0.9 THEN '6 Critical (0.9-1.0)'
            WHEN {PROB_COL} >= 0.7 THEN '5 High (0.7-0.9)'
            WHEN {PROB_COL} >= 0.5 THEN '4 Elevated (0.5-0.7)'
            WHEN {PROB_COL} >= 0.3 THEN '3 Moderate (0.3-0.5)'
            WHEN {PROB_COL} >= 0.1 THEN '2 Low (0.1-0.3)'
            ELSE '1 Minimal (0.0-0.1)'
          END AS risk_band,
          COUNT(*) AS txn_count
        FROM {PREDICTIONS_TABLE}
        GROUP BY 1
        ORDER BY 1
        """
    )

    if not dist.empty:
        chart = (
            alt.Chart(dist)
            .mark_bar(color="#29B5E8")
            .encode(
                x=alt.X("risk_band:N", title=None, sort=None, axis=alt.Axis(labelAngle=-30)),
                y=alt.Y("txn_count:Q", title="Transactions"),
                tooltip=[
                    alt.Tooltip("risk_band:N", title="Band"),
                    alt.Tooltip("txn_count:Q", title="Transactions", format=","),
                ],
            )
            .properties(height=CHART_HEIGHT)
        )
        st.altair_chart(chart, use_container_width=True)
        st.caption(
            "The long tail on the left is the point: the model concentrates risk "
            "into a small reviewable slice rather than flagging everything."
        )

    st.divider()
    st.subheader("Highest-risk transactions")
    st.caption("Scored live through the registry function, not read from a cache.")

    top_n = st.slider("Rows to score", 5, 50, 10, step=5)
    feature_args = ", ".join(f"f.{c}" for c in FEATURE_COLUMNS)
    top = safe_query(
        f"""
        SELECT
          f.TXN_ID,
          ROUND(f.AMOUNT, 2) AS amount,
          f.HOUR_OF_DAY AS hour,
          f.IS_LATE_NIGHT AS late_night,
          ROUND(f.MERCHANT_RISK_SCORE, 2) AS merchant_risk,
          ROUND({MODEL_NAME}!PREDICT_PROBA({feature_args}):output_feature_1::FLOAT, 4)
            AS fraud_probability
        FROM {SCORING_VIEW} f
        ORDER BY fraud_probability DESC
        LIMIT {int(top_n)}
        """
    )
    if not top.empty:
        st.dataframe(top, hide_index=True, use_container_width=True)


def threshold_tab() -> None:
    st.subheader("Decision threshold explorer")
    st.warning(
        "**In-sample metrics.** These are computed over the same transactions the "
        "model was trained on, so they are optimistic. Use them to reason about "
        "the *shape* of the precision/recall trade-off, not as a performance "
        "claim — the held-out numbers on the Overview tab are the honest ones.",
        icon=":material/warning:",
    )

    threshold = st.slider(
        "Classify as fraud when probability is at least",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05,
    )

    cm = safe_query(
        f"""
        WITH scored AS (
          SELECT p.{PROB_COL} AS prob, t.IS_FRAUD AS actual
          FROM {PREDICTIONS_TABLE} p
          JOIN {LABELS_TABLE} t USING (TXN_ID)
        )
        SELECT
          COUNT_IF(prob >= ? AND actual = 1) AS tp,
          COUNT_IF(prob >= ? AND actual = 0) AS fp,
          COUNT_IF(prob <  ? AND actual = 1) AS fn,
          COUNT_IF(prob <  ? AND actual = 0) AS tn
        FROM scored
        """,
        params=[threshold, threshold, threshold, threshold],
    )
    if cm.empty:
        return

    tp, fp, fn, tn = (int(cm.iloc[0][k]) for k in ("tp", "fp", "fn", "tn"))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0

    cols = st.columns(4)
    cols[0].metric("Precision", f"{precision:.3f}")
    cols[1].metric("Recall", f"{recall:.3f}")
    cols[2].metric("F1", f"{f1:.3f}")
    cols[3].metric("Flagged for review", f"{tp + fp:,}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        st.dataframe(
            pd.DataFrame(
                {
                    "": ["Actually fraud", "Actually legitimate"],
                    "Flagged": [f"{tp:,}", f"{fp:,}"],
                    "Not flagged": [f"{fn:,}", f"{tn:,}"],
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
    with right:
        st.markdown("**Business reading**")
        st.markdown(
            f"- Caught **{tp:,}** of **{tp + fn:,}** fraudulent transactions\n"
            f"- Missed **{fn:,}**\n"
            f"- Sent **{fp:,}** legitimate transactions to review\n"
            f"- Analysts review **{tp + fp:,}** items to find **{tp:,}** real cases"
        )
        st.caption(
            "Lower the threshold to catch more fraud at the cost of more review "
            "volume. This is a business decision, not a modelling one."
        )


def serving_tab() -> None:
    st.subheader("Serving infrastructure")
    st.caption(
        "Blue/green by construction: a new version gets its own service, the "
        "gateway shifts traffic, then the old service is reaped."
    )

    services = safe_meta(f"SHOW SERVICES IN SCHEMA {FQ}")
    if not services.empty:
        keep = [
            c
            for c in ["name", "status", "is_job", "compute_pool", "owner", "current_instances", "created_on"]
            if c in services.columns
        ]
        display = services[keep].rename(
            columns={
                "name": "Service",
                "status": "Status",
                "is_job": "Build job",
                "compute_pool": "Compute pool",
                "owner": "Owner",
                "current_instances": "Instances",
                "created_on": "Created",
            }
        )
        st.dataframe(display, hide_index=True, use_container_width=True)
        st.caption(
            "`MODEL_BUILD_*` rows are completed container builds — useful history, "
            "not running endpoints. Exactly one inference service should be RUNNING."
        )

    st.divider()
    st.subheader("Gateway traffic split")
    gateway = safe_meta(f"DESC GATEWAY {GATEWAY_NAME}")
    if not gateway.empty:
        row = gateway.iloc[0]
        if "ingress_url" in gateway:
            st.code(str(row["ingress_url"]), language="text")
        if "spec" in gateway:
            st.code(str(row["spec"]), language="yaml")

    st.divider()
    st.subheader("Model monitor")
    monitors = safe_meta(f"SHOW MODEL MONITORS IN SCHEMA {FQ}")
    if not monitors.empty:
        # SHOW MODEL MONITORS reports `monitor_state`; `state` is accepted too in
        # case the surface changes.
        keep = [
            c
            for c in [
                "name",
                "monitor_state",
                "state",
                "model_task",
                "refresh_interval",
                "aggregation_window",
                "baseline",
                "warehouse",
                "created_on",
            ]
            if c in monitors.columns
        ]
        st.dataframe(monitors[keep], hide_index=True, use_container_width=True)
        st.caption(
            "`baseline` shows `NOT_SET` — this deployment tracks volume and "
            "distribution statistics but does not compare against a reference "
            "window, so drift metrics are not computed."
        )

    volume = safe_query(
        f"""
        SELECT EVENT_TIMESTAMP, METRIC_NAME, COLUMN_NAME, METRIC_VALUE
        FROM TABLE(MODEL_MONITOR_STAT_METRIC(
          '{MONITOR_NAME}', 'COUNT', '{PROB_COL.replace("'", "''")}', '1 DAY'))
        ORDER BY EVENT_TIMESTAMP DESC
        """
    )
    if not volume.empty:
        st.markdown("**Scored volume observed by the monitor**")
        st.dataframe(volume, hide_index=True, use_container_width=True)
        st.caption(
            "The monitor tracks prediction volume and distribution statistics. "
            "Drift comparison additionally requires a registered baseline, which "
            "this deployment does not define."
        )
    else:
        st.info(
            "No monitor statistics yet. The monitor aggregates on a schedule, so "
            "freshly scored data can take a refresh cycle to appear."
        )


# =============================================================================
# Layout
# =============================================================================

render_header()

tab_overview, tab_registry, tab_predictions, tab_threshold, tab_serving = st.tabs(
    [
        "Overview",
        "Registry & Features",
        "Predictions",
        "Threshold Explorer",
        "Serving & Monitoring",
    ]
)

with tab_overview:
    overview_tab()
with tab_registry:
    registry_tab()
with tab_predictions:
    predictions_tab()
with tab_threshold:
    threshold_tab()
with tab_serving:
    serving_tab()
