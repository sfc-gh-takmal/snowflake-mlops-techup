#!/bin/bash
# Configure GitHub Actions CI/CD access to Snowflake
#
# This script:
#   1. Creates a network policy allowing GitHub-hosted runner IPs
#      (using Snowflake's managed network rule -- auto-tracks IP ranges)
#   2. Creates an authentication policy that permits WORKLOAD_IDENTITY
#   3. Creates MLOPS_DEPLOY_ROLE with least-privilege grants
#   4. Creates two service users with OIDC workload identity (STAGE + PROD)
#   5. Configures GitHub branch protection, environments, and repo variables
#
# Prerequisites:
#   - ACCOUNTADMIN role (creates roles, users, and account-level policies)
#   - scripts/setup.sh already run (the grants below reference its databases)
#   - gh CLI authenticated with admin access to the target repo
#
# Usage:
#   bash scripts/setup_cicd.sh
#
# Owner/repo are auto-detected from the current git remote. Override with:
#   GITHUB_OWNER=my-org GITHUB_REPO=my-repo bash scripts/setup_cicd.sh

set -euo pipefail

# --- CONFIGURATION ---------------------------------------------------------
# Resolve the target repo from the git remote unless explicitly overridden, then
# look up the numeric IDs via the API. GitHub enriches OIDC subject claims with
# these IDs (owner@<owner_id>/repo@<repo_id>), so they must be exact -- the old
# version of this script hardcoded another account's IDs, which silently
# produced subjects that could never match.
if [[ -z "${GITHUB_OWNER:-}" || -z "${GITHUB_REPO:-}" ]]; then
    if ! REPO_SLUG=$(gh repo view --json nameWithOwner --jq .nameWithOwner 2>/dev/null); then
        echo "ERROR: could not detect the GitHub repo." >&2
        echo "  Run this from inside the repo, or set GITHUB_OWNER and GITHUB_REPO." >&2
        exit 1
    fi
    GITHUB_OWNER="${REPO_SLUG%%/*}"
    GITHUB_REPO="${REPO_SLUG##*/}"
fi

echo "=== Resolving GitHub identifiers for ${GITHUB_OWNER}/${GITHUB_REPO} ==="
REPO_JSON=$(gh api "repos/${GITHUB_OWNER}/${GITHUB_REPO}")
GITHUB_OWNER_ID=$(echo "$REPO_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["owner"]["id"])')
GITHUB_REPO_ID=$(echo "$REPO_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')

if [[ -z "$GITHUB_OWNER_ID" || -z "$GITHUB_REPO_ID" ]]; then
    echo "ERROR: failed to resolve owner/repo IDs from the GitHub API." >&2
    exit 1
fi

REPO_REF="repo:${GITHUB_OWNER}@${GITHUB_OWNER_ID}/${GITHUB_REPO}@${GITHUB_REPO_ID}"
# One SUBJECT per user: WORKLOAD_IDENTITY accepts exactly one, so STAGE and PROD
# must be separate users keyed on their GitHub Environment.
STAGE_SUBJECT="${REPO_REF}:environment:STAGE"
PROD_SUBJECT="${REPO_REF}:environment:PROD"
SERVICE_USER_STAGE="SVC_GITHUB_ACTIONS_STAGE"
SERVICE_USER_PROD="SVC_GITHUB_ACTIONS"

# The auth policy lives outside the demo databases so that dropping
# SNOW_MLOPS_* during teardown does not fail on a policy reference.
POLICY_DB="MLOPS_CICD"
POLICY_SCHEMA="SECURITY"
AUTH_POLICY="${POLICY_DB}.${POLICY_SCHEMA}.MLOPS_OIDC_POLICY"

echo "=== Setting up GitHub Actions CI/CD access ==="
echo "  Repo:          ${GITHUB_OWNER}/${GITHUB_REPO} (owner_id=${GITHUB_OWNER_ID}, repo_id=${GITHUB_REPO_ID})"
echo "  STAGE user:    ${SERVICE_USER_STAGE}"
echo "  STAGE subject: ${STAGE_SUBJECT}"
echo "  PROD user:     ${SERVICE_USER_PROD}"
echo "  PROD subject:  ${PROD_SUBJECT}"
echo "  Auth policy:   ${AUTH_POLICY}"
echo ""

snow sql -q "
-- =============================================================================
-- Step 1: Network Policy for GitHub Actions
-- Uses Snowflake's managed network rule that auto-tracks GitHub runner IPs.
-- Note: do NOT hand-roll a rule here. MODE = INGRESS requires an IP-based TYPE,
-- so 'MODE = INGRESS TYPE = HOST_PORT' is invalid and fails to create.
-- =============================================================================
CREATE NETWORK POLICY IF NOT EXISTS GITHUB_ACTIONS_POLICY
    ALLOWED_NETWORK_RULE_LIST = ('SNOWFLAKE.NETWORK_SECURITY.GITHUBACTIONS_GLOBAL')
    COMMENT = 'Allow GitHub Actions runners via managed network rule';

-- =============================================================================
-- Step 2: Authentication Policy permitting WORKLOAD_IDENTITY
-- An account-level authentication policy that omits WORKLOAD_IDENTITY blocks all
-- OIDC logins before the subject is ever evaluated, surfacing as
-- 'Authentication attempt rejected by the current authentication policy'.
-- A user-level policy overrides the account default, so attach this to the
-- service users only rather than modifying a shared account policy.
-- =============================================================================
CREATE DATABASE IF NOT EXISTS ${POLICY_DB}
    COMMENT = 'Holds CI/CD security objects that must outlive SNOW_MLOPS_* teardown';
CREATE SCHEMA IF NOT EXISTS ${POLICY_DB}.${POLICY_SCHEMA};

CREATE AUTHENTICATION POLICY IF NOT EXISTS ${AUTH_POLICY}
    AUTHENTICATION_METHODS = ('OAUTH', 'WORKLOAD_IDENTITY')
    WORKLOAD_IDENTITY_POLICY = (
        ALLOWED_PROVIDERS = ('OIDC')
        ALLOWED_OIDC_ISSUERS = ('https://token.actions.githubusercontent.com')
    )
    COMMENT = 'Allows GitHub Actions OIDC workload identity for MLOps CI/CD service users';

-- =============================================================================
-- Step 3: Create MLOPS_DEPLOY_ROLE (least-privilege for CI/CD)
-- =============================================================================
CREATE ROLE IF NOT EXISTS MLOPS_DEPLOY_ROLE
    COMMENT = 'Least-privilege role for MLOps CI/CD deployments';

-- Database access
GRANT USAGE ON DATABASE SNOW_MLOPS_DEV TO ROLE MLOPS_DEPLOY_ROLE;
GRANT USAGE ON DATABASE SNOW_MLOPS_STAGE TO ROLE MLOPS_DEPLOY_ROLE;
GRANT USAGE ON DATABASE SNOW_MLOPS_PROD TO ROLE MLOPS_DEPLOY_ROLE;

-- Full schema control
GRANT ALL ON SCHEMA SNOW_MLOPS_DEV.ML TO ROLE MLOPS_DEPLOY_ROLE;
GRANT ALL ON SCHEMA SNOW_MLOPS_STAGE.ML TO ROLE MLOPS_DEPLOY_ROLE;
GRANT ALL ON SCHEMA SNOW_MLOPS_PROD.ML TO ROLE MLOPS_DEPLOY_ROLE;

-- Existing + future objects (tables, stages, dynamic tables)
GRANT ALL ON ALL TABLES IN SCHEMA SNOW_MLOPS_DEV.ML TO ROLE MLOPS_DEPLOY_ROLE;
GRANT ALL ON ALL TABLES IN SCHEMA SNOW_MLOPS_STAGE.ML TO ROLE MLOPS_DEPLOY_ROLE;
GRANT ALL ON ALL TABLES IN SCHEMA SNOW_MLOPS_PROD.ML TO ROLE MLOPS_DEPLOY_ROLE;
GRANT ALL ON FUTURE TABLES IN SCHEMA SNOW_MLOPS_DEV.ML TO ROLE MLOPS_DEPLOY_ROLE;
GRANT ALL ON FUTURE TABLES IN SCHEMA SNOW_MLOPS_STAGE.ML TO ROLE MLOPS_DEPLOY_ROLE;
GRANT ALL ON FUTURE TABLES IN SCHEMA SNOW_MLOPS_PROD.ML TO ROLE MLOPS_DEPLOY_ROLE;

-- The Feature Store issues CREATE OR REPLACE DYNAMIC TABLE, so the CI role needs
-- OWNERSHIP on feature-view dynamic tables, not just DML. FUTURE grants cover
-- tables created after this script runs.
GRANT ALL ON ALL DYNAMIC TABLES IN SCHEMA SNOW_MLOPS_DEV.ML TO ROLE MLOPS_DEPLOY_ROLE;
GRANT ALL ON ALL DYNAMIC TABLES IN SCHEMA SNOW_MLOPS_STAGE.ML TO ROLE MLOPS_DEPLOY_ROLE;
GRANT ALL ON ALL DYNAMIC TABLES IN SCHEMA SNOW_MLOPS_PROD.ML TO ROLE MLOPS_DEPLOY_ROLE;
GRANT ALL ON FUTURE DYNAMIC TABLES IN SCHEMA SNOW_MLOPS_DEV.ML TO ROLE MLOPS_DEPLOY_ROLE;
GRANT ALL ON FUTURE DYNAMIC TABLES IN SCHEMA SNOW_MLOPS_STAGE.ML TO ROLE MLOPS_DEPLOY_ROLE;
GRANT ALL ON FUTURE DYNAMIC TABLES IN SCHEMA SNOW_MLOPS_PROD.ML TO ROLE MLOPS_DEPLOY_ROLE;

GRANT ALL ON ALL STAGES IN SCHEMA SNOW_MLOPS_DEV.ML TO ROLE MLOPS_DEPLOY_ROLE;
GRANT ALL ON ALL STAGES IN SCHEMA SNOW_MLOPS_STAGE.ML TO ROLE MLOPS_DEPLOY_ROLE;
GRANT ALL ON ALL STAGES IN SCHEMA SNOW_MLOPS_PROD.ML TO ROLE MLOPS_DEPLOY_ROLE;
GRANT ALL ON FUTURE STAGES IN SCHEMA SNOW_MLOPS_DEV.ML TO ROLE MLOPS_DEPLOY_ROLE;
GRANT ALL ON FUTURE STAGES IN SCHEMA SNOW_MLOPS_STAGE.ML TO ROLE MLOPS_DEPLOY_ROLE;
GRANT ALL ON FUTURE STAGES IN SCHEMA SNOW_MLOPS_PROD.ML TO ROLE MLOPS_DEPLOY_ROLE;

-- Views (the Feature Store and monitors create views alongside dynamic tables)
GRANT ALL ON ALL VIEWS IN SCHEMA SNOW_MLOPS_DEV.ML TO ROLE MLOPS_DEPLOY_ROLE;
GRANT ALL ON ALL VIEWS IN SCHEMA SNOW_MLOPS_STAGE.ML TO ROLE MLOPS_DEPLOY_ROLE;
GRANT ALL ON ALL VIEWS IN SCHEMA SNOW_MLOPS_PROD.ML TO ROLE MLOPS_DEPLOY_ROLE;
GRANT ALL ON FUTURE VIEWS IN SCHEMA SNOW_MLOPS_DEV.ML TO ROLE MLOPS_DEPLOY_ROLE;
GRANT ALL ON FUTURE VIEWS IN SCHEMA SNOW_MLOPS_STAGE.ML TO ROLE MLOPS_DEPLOY_ROLE;
GRANT ALL ON FUTURE VIEWS IN SCHEMA SNOW_MLOPS_PROD.ML TO ROLE MLOPS_DEPLOY_ROLE;

-- Models: grant ownership on existing models so MLOPS_DEPLOY_ROLE can manage them
-- (Models created by ACCOUNTADMIN are otherwise inaccessible to other roles)
GRANT OWNERSHIP ON ALL MODELS IN SCHEMA SNOW_MLOPS_DEV.ML TO ROLE MLOPS_DEPLOY_ROLE COPY CURRENT GRANTS;
GRANT OWNERSHIP ON ALL MODELS IN SCHEMA SNOW_MLOPS_STAGE.ML TO ROLE MLOPS_DEPLOY_ROLE COPY CURRENT GRANTS;
GRANT OWNERSHIP ON ALL MODELS IN SCHEMA SNOW_MLOPS_PROD.ML TO ROLE MLOPS_DEPLOY_ROLE COPY CURRENT GRANTS;

-- OWNERSHIP (not just ALL) is required wherever the pipeline issues
-- CREATE OR REPLACE against an object that already exists:
--   * dynamic tables -- the Feature Store replaces feature views on every register
--   * views          -- FRAUD_SCORING_FEATURES is CREATE OR REPLACE VIEW
--   * tables         -- BATCH_PREDICTIONS is written with save_as_table(mode=overwrite)
-- GRANT ALL does NOT include OWNERSHIP, so without these a CI run against objects
-- first created by an interactive role fails on insufficient privileges.
GRANT OWNERSHIP ON ALL DYNAMIC TABLES IN SCHEMA SNOW_MLOPS_DEV.ML TO ROLE MLOPS_DEPLOY_ROLE COPY CURRENT GRANTS;
GRANT OWNERSHIP ON ALL DYNAMIC TABLES IN SCHEMA SNOW_MLOPS_STAGE.ML TO ROLE MLOPS_DEPLOY_ROLE COPY CURRENT GRANTS;
GRANT OWNERSHIP ON ALL DYNAMIC TABLES IN SCHEMA SNOW_MLOPS_PROD.ML TO ROLE MLOPS_DEPLOY_ROLE COPY CURRENT GRANTS;
GRANT OWNERSHIP ON ALL VIEWS IN SCHEMA SNOW_MLOPS_DEV.ML TO ROLE MLOPS_DEPLOY_ROLE COPY CURRENT GRANTS;
GRANT OWNERSHIP ON ALL VIEWS IN SCHEMA SNOW_MLOPS_STAGE.ML TO ROLE MLOPS_DEPLOY_ROLE COPY CURRENT GRANTS;
GRANT OWNERSHIP ON ALL VIEWS IN SCHEMA SNOW_MLOPS_PROD.ML TO ROLE MLOPS_DEPLOY_ROLE COPY CURRENT GRANTS;
GRANT OWNERSHIP ON ALL TABLES IN SCHEMA SNOW_MLOPS_DEV.ML TO ROLE MLOPS_DEPLOY_ROLE COPY CURRENT GRANTS;
GRANT OWNERSHIP ON ALL TABLES IN SCHEMA SNOW_MLOPS_STAGE.ML TO ROLE MLOPS_DEPLOY_ROLE COPY CURRENT GRANTS;
GRANT OWNERSHIP ON ALL TABLES IN SCHEMA SNOW_MLOPS_PROD.ML TO ROLE MLOPS_DEPLOY_ROLE COPY CURRENT GRANTS;

-- Tasks: the DAG is deployed with CreateMode.or_replace, which requires OWNERSHIP
-- on any pre-existing task. Without this a CI deploy against a DAG first created
-- interactively fails with:
--   task ... already exists, but current role has no privileges on it
-- Tasks must be suspended before ownership can move, so suspend the DAG first.
ALTER TASK IF EXISTS SNOW_MLOPS_DEV.ML.ML_TRAINING_PIPELINE SUSPEND;
ALTER TASK IF EXISTS SNOW_MLOPS_STAGE.ML.ML_TRAINING_PIPELINE SUSPEND;
ALTER TASK IF EXISTS SNOW_MLOPS_PROD.ML.ML_TRAINING_PIPELINE SUSPEND;
GRANT OWNERSHIP ON ALL TASKS IN SCHEMA SNOW_MLOPS_DEV.ML TO ROLE MLOPS_DEPLOY_ROLE COPY CURRENT GRANTS;
GRANT OWNERSHIP ON ALL TASKS IN SCHEMA SNOW_MLOPS_STAGE.ML TO ROLE MLOPS_DEPLOY_ROLE COPY CURRENT GRANTS;
GRANT OWNERSHIP ON ALL TASKS IN SCHEMA SNOW_MLOPS_PROD.ML TO ROLE MLOPS_DEPLOY_ROLE COPY CURRENT GRANTS;

-- Warehouses + Compute Pools
GRANT USAGE ON WAREHOUSE SNOW_MLOPS_DEV_WH TO ROLE MLOPS_DEPLOY_ROLE;
GRANT USAGE ON WAREHOUSE SNOW_MLOPS_STAGE_WH TO ROLE MLOPS_DEPLOY_ROLE;
GRANT USAGE ON WAREHOUSE SNOW_MLOPS_PROD_WH TO ROLE MLOPS_DEPLOY_ROLE;
GRANT USAGE, MONITOR ON COMPUTE POOL SNOW_MLOPS_DEV_POOL TO ROLE MLOPS_DEPLOY_ROLE;
GRANT USAGE, MONITOR ON COMPUTE POOL SNOW_MLOPS_STAGE_POOL TO ROLE MLOPS_DEPLOY_ROLE;
GRANT USAGE, MONITOR ON COMPUTE POOL SNOW_MLOPS_PROD_POOL TO ROLE MLOPS_DEPLOY_ROLE;

-- Account-level privileges
GRANT BIND SERVICE ENDPOINT ON ACCOUNT TO ROLE MLOPS_DEPLOY_ROLE;
GRANT EXECUTE TASK ON ACCOUNT TO ROLE MLOPS_DEPLOY_ROLE;
GRANT EXECUTE MANAGED TASK ON ACCOUNT TO ROLE MLOPS_DEPLOY_ROLE;

-- Admin can manage this role
GRANT ROLE MLOPS_DEPLOY_ROLE TO ROLE ACCOUNTADMIN;

-- =============================================================================
-- Step 4: STAGE Service User (runs on every merge to main)
-- =============================================================================
CREATE USER IF NOT EXISTS ${SERVICE_USER_STAGE}
    TYPE = SERVICE
    DEFAULT_ROLE = MLOPS_DEPLOY_ROLE
    COMMENT = 'GitHub Actions CI/CD service user for STAGE (OIDC)'
    WORKLOAD_IDENTITY = (
        TYPE = OIDC
        ISSUER = 'https://token.actions.githubusercontent.com'
        SUBJECT = '${STAGE_SUBJECT}'
    );

-- Re-apply the subject explicitly so re-runs after a repo change take effect
-- (CREATE USER IF NOT EXISTS is a no-op on an existing user).
ALTER USER ${SERVICE_USER_STAGE} SET WORKLOAD_IDENTITY = (
    TYPE = OIDC
    ISSUER = 'https://token.actions.githubusercontent.com'
    SUBJECT = '${STAGE_SUBJECT}'
);

GRANT ROLE MLOPS_DEPLOY_ROLE TO USER ${SERVICE_USER_STAGE};
ALTER USER ${SERVICE_USER_STAGE} SET NETWORK_POLICY = 'GITHUB_ACTIONS_POLICY';

-- =============================================================================
-- Step 5: PROD Service User (gated by the PROD environment reviewer)
-- =============================================================================
CREATE USER IF NOT EXISTS ${SERVICE_USER_PROD}
    TYPE = SERVICE
    DEFAULT_ROLE = MLOPS_DEPLOY_ROLE
    COMMENT = 'GitHub Actions CI/CD service user for PROD (OIDC)'
    WORKLOAD_IDENTITY = (
        TYPE = OIDC
        ISSUER = 'https://token.actions.githubusercontent.com'
        SUBJECT = '${PROD_SUBJECT}'
    );

ALTER USER ${SERVICE_USER_PROD} SET WORKLOAD_IDENTITY = (
    TYPE = OIDC
    ISSUER = 'https://token.actions.githubusercontent.com'
    SUBJECT = '${PROD_SUBJECT}'
);

GRANT ROLE MLOPS_DEPLOY_ROLE TO USER ${SERVICE_USER_PROD};
ALTER USER ${SERVICE_USER_PROD} SET NETWORK_POLICY = 'GITHUB_ACTIONS_POLICY';
"

echo ""
echo "=== Snowflake CI/CD access configured ==="

# =============================================================================
# Step 5a: Attach the authentication policy
#
# A user can hold only one AUTHENTICATION POLICY, and SET fails outright if one is
# already attached -- so this is not idempotent inline and has to UNSET first.
# UNSET is tolerated failing when nothing is attached (first run).
# =============================================================================
echo ""
echo "=== Attaching authentication policy ==="
for svc_user in "$SERVICE_USER_STAGE" "$SERVICE_USER_PROD"; do
    snow sql -q "ALTER USER ${svc_user} UNSET AUTHENTICATION POLICY" >/dev/null 2>&1 || true
    if snow sql -q "ALTER USER ${svc_user} SET AUTHENTICATION POLICY ${AUTH_POLICY}" >/dev/null 2>&1; then
        echo "  ${svc_user} -> ${AUTH_POLICY}"
    else
        echo "  ERROR: failed to attach ${AUTH_POLICY} to ${svc_user}" >&2
        echo "         OIDC logins will be rejected if an account-level policy omits" >&2
        echo "         WORKLOAD_IDENTITY. Investigate before running CI." >&2
        exit 1
    fi
done

# =============================================================================
# Step 5b: APPLY on Feature Store tags
#
# The Feature Store stamps metadata tags onto each feature-view dynamic table
# (SNOWML_FEATURE_STORE_OBJECT, SNOWML_FEATURE_VIEW_METADATA,
# SNOWML_FEATURE_STORE_ENTITY_<ENTITY>). Re-registering a feature view re-applies
# them, so the CI role needs APPLY on every one or the job fails with:
#   Insufficient privileges to operate on tag 'SNOWML_FEATURE_VIEW_METADATA'
#
# There is no bulk "GRANT APPLY ON ALL TAGS IN SCHEMA" form, so the tags have to be
# enumerated and granted individually. Entity tag names depend on the entities
# defined, so discover them rather than hardcoding.
# =============================================================================
echo ""
echo "=== Granting APPLY on Feature Store tags ==="

for db in SNOW_MLOPS_DEV SNOW_MLOPS_STAGE SNOW_MLOPS_PROD; do
    tags=$(snow sql -q "SHOW TAGS IN SCHEMA ${db}.ML" --format json 2>/dev/null |
        python3 -c 'import json,sys
try:
    rows = json.load(sys.stdin)
except Exception:
    rows = []
print("\n".join(r["name"] for r in rows))' 2>/dev/null)

    if [[ -z "$tags" ]]; then
        echo "  ${db}.ML: no tags yet (CI will create and own them)"
        continue
    fi

    while IFS= read -r tag; do
        [[ -z "$tag" ]] && continue
        snow sql -q "GRANT APPLY ON TAG ${db}.ML.\"${tag}\" TO ROLE MLOPS_DEPLOY_ROLE" >/dev/null 2>&1 &&
            echo "  APPLY granted: ${db}.ML.${tag}" ||
            echo "  WARNING: could not grant APPLY on ${db}.ML.${tag}" >&2
    done <<< "$tags"
done

# =============================================================================
# Step 5c: OWNERSHIP on Experiments
#
# Experiment Tracking calls set_experiment(), which fails on an experiment owned by
# another role:
#   Experiment 'FRAUD_DETECTION_TRAINING' already exists, but current role has no
#   privileges on it
#
# As with tags there is no bulk form -- "GRANT on all objects of type EXPERIMENT"
# returns Unsupported feature -- so enumerate and grant individually.
# =============================================================================
echo ""
echo "=== Granting OWNERSHIP on Experiments ==="

for db in SNOW_MLOPS_DEV SNOW_MLOPS_STAGE SNOW_MLOPS_PROD; do
    experiments=$(snow sql -q "SHOW EXPERIMENTS IN SCHEMA ${db}.ML" --format json 2>/dev/null |
        python3 -c 'import json,sys
try:
    rows = json.load(sys.stdin)
except Exception:
    rows = []
print("\n".join(r["name"] for r in rows))' 2>/dev/null)

    if [[ -z "$experiments" ]]; then
        echo "  ${db}.ML: no experiments yet (CI will create and own them)"
        continue
    fi

    while IFS= read -r exp; do
        [[ -z "$exp" ]] && continue
        snow sql -q "GRANT OWNERSHIP ON EXPERIMENT ${db}.ML.\"${exp}\" TO ROLE MLOPS_DEPLOY_ROLE COPY CURRENT GRANTS" \
            >/dev/null 2>&1 &&
            echo "  OWNERSHIP granted: ${db}.ML.${exp}" ||
            echo "  WARNING: could not grant OWNERSHIP on experiment ${db}.ML.${exp}" >&2
    done <<< "$experiments"
done

# =============================================================================
# Step 6: GitHub repo variables
# The workflows read these; nothing here is a secret (auth is keyless OIDC).
# =============================================================================
echo ""
echo "=== Setting GitHub repo variables ==="

SNOWFLAKE_ACCOUNT_VALUE="${SNOWFLAKE_ACCOUNT:-$(snow sql -q 'SELECT CURRENT_ACCOUNT()' --format json |
    python3 -c 'import json,sys; print(list(json.load(sys.stdin)[0].values())[0])')}"

set_var() {
    gh variable set "$1" --body "$2" --repo "${GITHUB_OWNER}/${GITHUB_REPO}" >/dev/null
    echo "  ${1} = ${2}"
}

set_var SNOWFLAKE_ACCOUNT "$SNOWFLAKE_ACCOUNT_VALUE"
set_var SNOWFLAKE_DATABASE_STAGE "SNOW_MLOPS_STAGE"
set_var SNOWFLAKE_DATABASE_PROD "SNOW_MLOPS_PROD"
set_var SNOWFLAKE_SCHEMA "ML"
set_var SNOWFLAKE_USER_STAGE "$SERVICE_USER_STAGE"
set_var SNOWFLAKE_USER_PROD "$SERVICE_USER_PROD"
set_var TOPOLOGY "single-account"
set_var ENABLE_MODEL_MONITOR "true"

# =============================================================================
# Step 7: GitHub Environments
# The PROD approval gate is enforced entirely by the environment protection rule.
# Required reviewers need a plan that supports them on this repo's visibility;
# if that call fails we surface a loud warning rather than silently continuing
# with an unprotected PROD environment.
# =============================================================================
echo ""
echo "=== Creating GitHub Environments ==="

gh api "repos/${GITHUB_OWNER}/${GITHUB_REPO}/environments/STAGE" -X PUT --input - <<<'{}' >/dev/null
echo "  STAGE environment created (no protection rules)"

REVIEWER_ID=$(gh api user --jq '.id')
if gh api "repos/${GITHUB_OWNER}/${GITHUB_REPO}/environments/PROD" -X PUT --input - >/dev/null 2>&1 <<EOF
{"reviewers":[{"type":"User","id":${REVIEWER_ID}}]}
EOF
then
    echo "  PROD environment created with required reviewer (user id ${REVIEWER_ID})"
else
    gh api "repos/${GITHUB_OWNER}/${GITHUB_REPO}/environments/PROD" -X PUT --input - <<<'{}' >/dev/null
    echo "  WARNING: PROD environment created WITHOUT a required reviewer." >&2
    echo "           GitHub rejected the reviewer rule (usually a plan/visibility limit)." >&2
    echo "           The deploy-prod job will NOT pause for approval until this is fixed" >&2
    echo "           under Settings > Environments > PROD > Required reviewers." >&2
fi

# =============================================================================
# Step 8: Branch protection
#
# Required status checks + enforce_admins already prevent direct pushes to main,
# so every change must go through a PR whose lint and test checks pass.
#
# PR *review* approval is off by default (REQUIRED_REVIEWS=0). GitHub does not
# allow approving your own pull request, so on a single-operator demo repo a
# review requirement makes main unmergeable with no way to self-serve. The
# human-approval story lives on the PROD environment reviewer gate instead,
# which *can* be self-approved.
#
# Set REQUIRED_REVIEWS=1 on a repo with more than one maintainer.
#
# Best-effort: unavailable on some plans for private repos.
# =============================================================================
echo ""
echo "=== Setting up GitHub branch protection ==="

REQUIRED_REVIEWS="${REQUIRED_REVIEWS:-0}"
if [[ "$REQUIRED_REVIEWS" -gt 0 ]]; then
    reviews_json="{\"required_approving_review_count\":${REQUIRED_REVIEWS},\"dismiss_stale_reviews\":true}"
    echo "  Requiring ${REQUIRED_REVIEWS} approving review(s)"
else
    reviews_json="null"
    echo "  PR reviews not required (single-operator mode; PROD environment gate still applies)"
fi

if gh api "repos/${GITHUB_OWNER}/${GITHUB_REPO}/branches/main/protection" -X PUT --input - >/dev/null 2>&1 <<PROTECTION
{
  "required_status_checks": {
    "strict": true,
    "contexts": ["lint", "test"]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": ${reviews_json},
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false
}
PROTECTION
then
    echo "  - Direct pushes to main blocked (admins included)"
    echo "  - Status checks 'lint' and 'test' must pass"
    echo "  - Force pushes and branch deletion blocked"
else
    echo "  WARNING: branch protection could not be applied (plan or permission limit)." >&2
    echo "           Direct pushes to main will not be blocked." >&2
fi

echo ""
echo "=== CI/CD setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Push to main to trigger the deploy workflow:"
echo "       gh workflow run deploy.yml --repo ${GITHUB_OWNER}/${GITHUB_REPO}"
echo "  2. Watch it: gh run watch --repo ${GITHUB_OWNER}/${GITHUB_REPO}"
echo "  3. Approve the PROD deployment when the workflow pauses."
echo ""
echo "If 'snow connection test' fails in CI with 'Authentication attempt rejected"
echo "by the current authentication policy', an account-level authentication policy"
echo "is blocking WORKLOAD_IDENTITY. Verify with:"
echo "  snow sql -q \"DESCRIBE AUTHENTICATION POLICY ${AUTH_POLICY}\""
echo "If the subject does not match, the Actions log prints the actual claim --"
echo "compare it against: ${STAGE_SUBJECT}"
