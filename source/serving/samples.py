"""Build model input samples that are guaranteed to match the training signature.

Hand-written sample dicts were previously duplicated in deploy_prod_service.py and
tests/test_endpoint.py, with numpy dtypes chosen by hand (np.int8, np.int16, ...).
Any drift between those literals and the real feature table produced signature
mismatches at inference time.

Taking a real row from the scoring view instead means the column set, order and
dtypes always match whatever the model was trained on.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FEATURE_COLUMNS, SCORING_VIEW_NAME


def scoring_view_fqn(database: str, schema: str) -> str:
    return f"{database}.{schema}.{SCORING_VIEW_NAME}"


def get_sample(session, database: str, schema: str, n: int = 1):
    """Return `n` real rows from the scoring view as a pandas DataFrame.

    Only the model's feature columns are selected, in FEATURE_COLUMNS order.
    """
    cols = ", ".join(FEATURE_COLUMNS)
    df = session.sql(f"SELECT {cols} FROM {scoring_view_fqn(database, schema)} LIMIT {n}").to_pandas()
    if df.empty:
        raise RuntimeError(f"{scoring_view_fqn(database, schema)} returned no rows -- cannot build a sample")
    return df[FEATURE_COLUMNS]


def get_contrasting_samples(session, database: str, schema: str):
    """Return (low_risk_row, high_risk_row) drawn from real data.

    Picks the transaction with the smallest and largest combination of the signals
    the generator actually used to inject fraud (amount ratio, merchant risk,
    late-night). Used to assert the model orders risk sensibly without hardcoding
    feature values.
    """
    cols = ", ".join(FEATURE_COLUMNS)
    fqn = scoring_view_fqn(database, schema)

    low = session.sql(f"""
        SELECT {cols} FROM {fqn}
        WHERE IS_LATE_NIGHT = 0 AND IS_HIGH_RISK_MERCHANT = 0
        ORDER BY AMOUNT_TO_AVG_RATIO ASC
        LIMIT 1
    """).to_pandas()

    high = session.sql(f"""
        SELECT {cols} FROM {fqn}
        WHERE IS_LATE_NIGHT = 1 AND IS_HIGH_RISK_MERCHANT = 1
        ORDER BY AMOUNT_TO_AVG_RATIO DESC
        LIMIT 1
    """).to_pandas()

    if low.empty or high.empty:
        raise RuntimeError(f"Could not find contrasting rows in {fqn}")

    return low[FEATURE_COLUMNS], high[FEATURE_COLUMNS]
