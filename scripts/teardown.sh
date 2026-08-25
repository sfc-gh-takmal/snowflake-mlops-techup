#!/bin/bash
# Tear down the Snowflake MLOps demo so it can be rebuilt from scratch.
#
# Order matters:
#   1. Drop the SPCS service and gateway first. TRANSFER OWNERSHIP ON SERVICE is
#      unsupported, so a service created by MLOPS_DEPLOY_ROLE must be dropped by
#      a role that inherits it (ACCOUNTADMIN does).
#   2. Detach the authentication policy from the service users. If the policy
#      lives inside a SNOW_MLOPS_* database, DROP DATABASE fails with
#      "Cannot drop database because policy '...' is set on user '...'".
#   3. Drop the databases (this cascades models, tasks, stages, monitors).
#
# By default this preserves warehouses, compute pools, MLOPS_DEPLOY_ROLE, and the
# MLOPS_CICD policy database -- setup.sh and setup_cicd.sh are idempotent and
# will adopt them. Pass --all to remove those too.
#
# Usage:
#   bash scripts/teardown.sh          # drop databases + serving objects
#   bash scripts/teardown.sh --all    # also drop pools, warehouses, role, policy db

set -euo pipefail

DROP_ALL=false
[[ "${1:-}" == "--all" ]] && DROP_ALL=true

echo "=== Tearing down Snowflake MLOps demo ==="
$DROP_ALL && echo "  Mode: FULL (databases + pools + warehouses + role)" \
          || echo "  Mode: databases only (pools, warehouses, role preserved)"
echo ""

echo "--- Step 1: drop serving objects (service, gateway) ---"
snow sql -q "
DROP SERVICE IF EXISTS SNOW_MLOPS_PROD.ML.MLOPS_FRAUD_DETECTOR_SERVICE_V1;
DROP SERVICE IF EXISTS SNOW_MLOPS_PROD.ML.MLOPS_FRAUD_DETECTOR_SERVICE_V2;
DROP SERVICE IF EXISTS SNOW_MLOPS_PROD.ML.MLOPS_FRAUD_DETECTOR_SERVICE_V3;
DROP SERVICE IF EXISTS SNOW_MLOPS_PROD.ML.MLOPS_FRAUD_DETECTOR_SERVICE_V4;
DROP SERVICE IF EXISTS SNOW_MLOPS_PROD.ML.MLOPS_FRAUD_DETECTOR_SERVICE_V5;
DROP GATEWAY IF EXISTS SNOW_MLOPS_PROD.ML.FRAUD_DETECTOR_GATEWAY;
" 2>&1 | grep -v '^$' || true

echo ""
echo "--- Step 2: detach authentication policy from service users ---"
# UNSET is only valid if the user exists and has a policy attached; tolerate both.
for user in SVC_GITHUB_ACTIONS_STAGE SVC_GITHUB_ACTIONS; do
    snow sql -q "ALTER USER IF EXISTS ${user} UNSET AUTHENTICATION POLICY" >/dev/null 2>&1 \
        && echo "  Detached policy from ${user}" \
        || echo "  ${user}: no policy attached (or user absent)"
done

echo ""
echo "--- Step 3: drop databases ---"
snow sql -q "
DROP DATABASE IF EXISTS SNOW_MLOPS_DEV;
DROP DATABASE IF EXISTS SNOW_MLOPS_STAGE;
DROP DATABASE IF EXISTS SNOW_MLOPS_PROD;
" 2>&1 | grep -v '^$' || true

echo ""
echo "--- Step 4: suspend compute pools (stop node billing) ---"
snow sql -q "
ALTER COMPUTE POOL IF EXISTS SNOW_MLOPS_DEV_POOL SUSPEND;
ALTER COMPUTE POOL IF EXISTS SNOW_MLOPS_STAGE_POOL SUSPEND;
ALTER COMPUTE POOL IF EXISTS SNOW_MLOPS_PROD_POOL SUSPEND;
" >/dev/null 2>&1 || true
echo "  Pools suspended"

if $DROP_ALL; then
    echo ""
    echo "--- Step 5: drop pools, warehouses, service users, role, policy db ---"
    snow sql -q "
DROP COMPUTE POOL IF EXISTS SNOW_MLOPS_DEV_POOL;
DROP COMPUTE POOL IF EXISTS SNOW_MLOPS_STAGE_POOL;
DROP COMPUTE POOL IF EXISTS SNOW_MLOPS_PROD_POOL;
DROP WAREHOUSE IF EXISTS SNOW_MLOPS_DEV_WH;
DROP WAREHOUSE IF EXISTS SNOW_MLOPS_STAGE_WH;
DROP WAREHOUSE IF EXISTS SNOW_MLOPS_PROD_WH;
DROP USER IF EXISTS SVC_GITHUB_ACTIONS_STAGE;
DROP USER IF EXISTS SVC_GITHUB_ACTIONS;
DROP ROLE IF EXISTS MLOPS_DEPLOY_ROLE;
DROP DATABASE IF EXISTS MLOPS_CICD;
" 2>&1 | grep -v '^$' || true
fi

echo ""
echo "--- Verifying teardown ---"
snow sql -q "SHOW DATABASES LIKE 'SNOW_MLOPS%'" --format json |
    python3 -c '
import json, sys
rows = json.load(sys.stdin)
if rows:
    print("  WARNING: databases still present: " + ", ".join(r["name"] for r in rows))
    sys.exit(1)
print("  OK: no SNOW_MLOPS_* databases remain")
'

echo ""
echo "=== Teardown complete. Rebuild with: bash scripts/setup.sh ==="
