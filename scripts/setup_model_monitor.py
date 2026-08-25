"""Set up Model Monitor — CI entry point.

Creates or updates the model monitor in the target environment.
Called by deploy-prod after batch inference has populated BATCH_PREDICTIONS.

Usage:
    uv run python scripts/setup_model_monitor.py [--env prod]
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "source"))
from monitoring.model_monitor import create_monitor, get_monitor_status
from snowpark_session import create_snowpark_session


def main():
    parser = argparse.ArgumentParser(description="Set up model monitor")
    parser.add_argument("--env", default=os.getenv("ML_ENV", "prod"), choices=["dev", "stage", "prod"])
    parser.add_argument("--validate-only", action="store_true", help="Create, verify, then drop (for STAGE validation)")
    args = parser.parse_args()

    print("=" * 60)
    if args.validate_only:
        print("MODEL MONITOR VALIDATION (create → verify → drop)")
    else:
        print("MODEL MONITOR SETUP")
    print("=" * 60)

    session = create_snowpark_session()

    # Create or update monitor
    print(f"\n[1/2] Creating/updating monitor (env={args.env})...")
    monitor_name = create_monitor(session, env=args.env)

    # Verify status
    print("\n[2/2] Verifying monitor status...")
    status = get_monitor_status(session, env=args.env)
    print(f"  Monitor: {monitor_name}")
    print(f"  State: {status.get('state', 'UNKNOWN')}")
    print(f"  Model version: {status.get('model_version', 'N/A')}")
    print(f"  Refresh interval: {status.get('refresh_interval', 'N/A')}")

    # Validate-only: drop after verification (STAGE validation)
    if args.validate_only:
        from monitoring.model_monitor import get_env_config

        cfg = get_env_config(args.env)
        db = os.getenv("SNOWFLAKE_DATABASE", cfg["database"])
        schema = os.getenv("SNOWFLAKE_SCHEMA", "ML")
        session.sql(f"DROP MODEL MONITOR IF EXISTS {db}.{schema}.{monitor_name}").collect()
        print("\n  Monitor dropped (validate-only mode).")
        print("\n" + "=" * 60)
        print("MODEL MONITOR VALIDATION PASSED")
        print("=" * 60)
        session.close()
        return

    # Write to Job Summary
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write("\n## Model Monitor\n\n")
            f.write("| Property | Value |\n|----------|-------|\n")
            f.write(f"| Monitor | `{monitor_name}` |\n")
            f.write(f"| State | {status.get('state', 'UNKNOWN')} |\n")
            f.write(f"| Model version | {status.get('model_version', 'N/A')} |\n")
            f.write(f"| Refresh interval | {status.get('refresh_interval', 'N/A')} |\n")

    print("\n" + "=" * 60)
    print("MODEL MONITOR ACTIVE")
    print("=" * 60)

    session.close()


if __name__ == "__main__":
    main()
