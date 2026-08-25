"""Model promotion dispatcher.

Routes model promotion to the correct strategy based on the TOPOLOGY
environment variable (or --strategy CLI arg).

Supported topologies:
  - single-account (default): Cross-database replication within one account
  - multi-account: Cross-account model sharing (same region)
  - cross-region: Cross-account + cross-region replication

Usage:
  python deploy/promote.py --version V3
  python deploy/promote.py --version V3 --strategy multi-account
  TOPOLOGY=single-account python deploy/promote.py --version V3
"""

import argparse
import os
import sys
from pathlib import Path

# When run as a script (python deploy/promote.py) Python puts deploy/ on sys.path,
# not the repo root, so `import deploy.strategies...` fails with ModuleNotFoundError.
# Add the repo root so the package imports resolve either way.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


STRATEGIES = {
    "single-account": "deploy.strategies.single_account",
    "multi-account": "deploy.strategies.multi_account",
    "cross-region": "deploy.strategies.cross_region",
}

DEFAULT_TOPOLOGY = "single-account"


def get_strategy(topology: str):
    """Dynamically import and return the promote function for a topology."""
    if topology not in STRATEGIES:
        print(f"ERROR: Unknown topology '{topology}'")
        print(f"  Available: {', '.join(STRATEGIES.keys())}")
        sys.exit(1)

    module_path = STRATEGIES[topology]
    # Import the module
    import importlib

    module = importlib.import_module(module_path)
    return module.promote


def main():
    parser = argparse.ArgumentParser(description="Promote a model version to PROD")
    parser.add_argument("--version", help="Model version to promote (e.g., V3)")
    parser.add_argument(
        "--latest-from-stage",
        action="store_true",
        help="Promote the highest V<n> currently in the STAGE registry",
    )
    parser.add_argument(
        "--strategy",
        default=os.getenv("TOPOLOGY", DEFAULT_TOPOLOGY),
        choices=STRATEGIES.keys(),
        help=f"Promotion topology (default: $TOPOLOGY or '{DEFAULT_TOPOLOGY}')",
    )
    args = parser.parse_args()

    if not args.version and not args.latest_from_stage:
        parser.error("provide --version V<n> or --latest-from-stage")

    version = args.version
    session = None
    if args.latest_from_stage:
        sys.path.insert(0, str(_REPO_ROOT / "source"))
        from config import MODEL_NAME
        from snowpark_session import create_snowpark_session

        from deploy.strategies.single_account import STAGE_DATABASE, STAGE_SCHEMA

        session = create_snowpark_session()
        rows = session.sql(f"SHOW VERSIONS IN MODEL {STAGE_DATABASE}.{STAGE_SCHEMA}.{MODEL_NAME}").collect()
        candidates = [r["name"] for r in rows if r["name"].startswith("V") and r["name"][1:].isdigit()]
        if not candidates:
            print(f"ERROR: no V<n> versions found in {STAGE_DATABASE}.{STAGE_SCHEMA}.{MODEL_NAME}")
            sys.exit(1)
        version = f"V{max(int(c[1:]) for c in candidates)}"

    print("Model Promotion")
    print(f"  Version:  {version}")
    print(f"  Strategy: {args.strategy}")
    print()

    promote_fn = get_strategy(args.strategy)
    promote_fn(version=version, session=session)

    print("\nPromotion complete.")


if __name__ == "__main__":
    main()
