#!/usr/bin/env python
"""Write the metrics files used by the quality-gate demo.

The quality gate reads its metrics from a JSON file (``METRICS_FILE``) and
exits *before* opening a Snowflake session, so both the passing and failing
paths run offline in well under a second. That makes the gate the one piece
of the pipeline that is genuinely demoable live.

This script materialises two fixtures so the demo needs no on-the-fly editing:

    demo_good.json      metrics from the real PROD model -- clears every gate
    demo_degraded.json  a plausibly bad model -- trips all three gates

Usage::

    uv run python scripts/seed_demo_metrics.py
    METRICS_FILE=/tmp/demo_good.json     uv run python scripts/quality_gate_and_register.py
    METRICS_FILE=/tmp/demo_degraded.json uv run python scripts/quality_gate_and_register.py

The degraded numbers are chosen to sit *below* every threshold in
``source/config.py`` so the gate reports three distinct failures rather than
one -- it reads better on a projector, and it shows the gate evaluates all
metrics instead of short-circuiting on the first miss.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from source.config import MIN_AUC_ROC, MIN_PRECISION, MIN_RECALL  # noqa: E402

# Actual held-out test metrics from the model currently serving in PROD
# (SNOW_MLOPS_PROD.ML.MLOPS_FRAUD_DETECTOR version V4).
GOOD_METRICS = {
    "auc_roc": 0.8952485395189003,
    "cv_auc_mean": 0.8871418062714775,
    "f1": 0.30319148936170215,
    "feature_view": "FRAUD_SCORING_FEATURES",
    "pr_auc": 0.4080893188935584,
    "precision": 0.19635826771653545,
    "recall": 0.665,
}

# A model that has genuinely regressed: barely better than coin-flip ranking,
# and a precision low enough that analysts would drown in false positives.
DEGRADED_METRICS = {
    "auc_roc": 0.62,
    "cv_auc_mean": 0.6104,
    "f1": 0.0709,
    "feature_view": "FRAUD_SCORING_FEATURES",
    "pr_auc": 0.0812,
    "precision": 0.04,
    "recall": 0.31,
}


def _assert_fixtures_are_valid() -> None:
    """Guard against config drift silently breaking the demo.

    If someone retunes the thresholds in ``source/config.py``, the fixtures
    can quietly stop demonstrating what they claim to. Fail loudly here
    instead of on stage.
    """
    for label, key, threshold in (
        ("AUC-ROC", "auc_roc", MIN_AUC_ROC),
        ("Precision", "precision", MIN_PRECISION),
        ("Recall", "recall", MIN_RECALL),
    ):
        if GOOD_METRICS[key] < threshold:
            raise SystemExit(
                f"demo_good.json would FAIL the gate: {label} "
                f"{GOOD_METRICS[key]} < {threshold}. "
                "Refresh GOOD_METRICS from the current PROD model."
            )
        if DEGRADED_METRICS[key] >= threshold:
            raise SystemExit(
                f"demo_degraded.json would PASS the {label} gate: "
                f"{DEGRADED_METRICS[key]} >= {threshold}. "
                "Lower DEGRADED_METRICS so the failure is unambiguous."
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default=os.getenv("DEMO_METRICS_DIR", "/tmp"),
        help="Directory to write the fixtures into (default: /tmp)",
    )
    args = parser.parse_args()

    _assert_fixtures_are_valid()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, payload in (
        ("demo_good.json", GOOD_METRICS),
        ("demo_degraded.json", DEGRADED_METRICS),
    ):
        path = out_dir / name
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"  wrote {path}")

    print(f"\nThresholds in effect: AUC-ROC >= {MIN_AUC_ROC}, Precision >= {MIN_PRECISION}, Recall >= {MIN_RECALL}")
    print("\nDemo the gate with:")
    print(f"  METRICS_FILE={out_dir / 'demo_good.json'} uv run python scripts/quality_gate_and_register.py")
    print(f"  METRICS_FILE={out_dir / 'demo_degraded.json'} uv run python scripts/quality_gate_and_register.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
