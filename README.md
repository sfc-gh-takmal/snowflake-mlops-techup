# Snowflake MLOps Template

An open-source, production-ready template for building end-to-end ML pipelines on Snowflake. This framework provides data science teams with starter code they can plug their own model logic into, while giving ML engineering teams a standardized process for training, validating, deploying, and monitoring models — all orchestrated through Git workflows with human-in-the-loop approvals.

Use this template as a bootstrap for your MLOps workflows, or as a baseline to extend with [Cortex Code](https://docs.snowflake.com/en/user-guide/cortex-code/cortex-code) for more customized agentic ML and MLOps workflows.

## What This Template Covers

- **Multi-step ML Pipelines** — Snowflake Task DAG (Python SDK) with configurable compute per step (warehouse or SPCS compute pools)
- **Feature Engineering** — Snowflake Feature Store with Dynamic Tables and configurable refresh intervals
- **Model Training** — ML Jobs (`@remote`) running XGBoost on compute pools with experiment tracking
- **Quality Gates** — Automated metric thresholds (AUC-ROC, precision, recall) that block bad models from promotion
- **Model Versioning** — Auto-incremented versions in the Snowflake Model Registry with full lineage
- **CI/CD Workflows** — GitHub Actions with OIDC auth, PR checks, and environment-gated approvals
- **Model Promotion** — Two paths: code-driven (push to main) and data-driven (scheduled retrain with candidate review)
- **Batch Inference** — Model Registry `run()` on warehouse with prediction validation
- **Real-Time Inference** — SPCS containers with blue/green Gateway deployment and zero-downtime rollouts
- **Model Monitoring** — Snowflake ML Observability tracking prediction drift and feature distribution shifts
- **Rollback** — One-click revert to any previous model version
- **Scheduled Retraining** — Weekly cron with GitHub Issue notifications for human review

## The Model

Per-transaction fraud detection on 100K synthetic transactions (3% fraud rate).

The model predicts `IS_FRAUD` for a single transaction using 17 numeric features that
combine transaction context with customer history, joined at `TXN_ID` granularity via
the `FRAUD_SCORING_FEATURES` view:

| Source | Features |
|--------|----------|
| Transaction context (`TRANSACTION_CONTEXT_FEATURES$V1`) | `AMOUNT`, `AMOUNT_TO_AVG_RATIO`, `IS_HIGH_RISK_MERCHANT`, `MERCHANT_RISK_SCORE`, `HOUR_OF_DAY`, `IS_WEEKEND`, `IS_LATE_NIGHT` |
| Customer history (`CUSTOMER_RISK_FEATURES$V1`) | `TOTAL_TXN_COUNT`, `AVG_TXN_AMOUNT`, `MAX_TXN_AMOUNT`, `STDDEV_TXN_AMOUNT`, `UNIQUE_MERCHANTS`, `ACTIVE_DAYS`, `LATE_NIGHT_TXN_RATIO`, `CREDIT_SCORE`, `ACCOUNT_AGE_DAYS`, `ANNUAL_INCOME` |

Observed performance on the synthetic dataset:

| Metric | Value | Note |
|--------|-------|------|
| AUC-ROC | 0.895 | CV mean 0.887 — no overfit |
| PR-AUC | 0.408 | 13.6x lift over the 3% base rate |
| Recall | 0.665 | catches two thirds of fraud |
| Precision | 0.196 | 6.5x the base rate |

### Two things to know about the feature set

**`HISTORICAL_FRAUD_COUNT` and `HISTORICAL_FRAUD_RATE` are deliberately excluded.**
They exist on the customer feature view, but they are aggregates of `IS_FRAUD` over the
same transactions being scored. Including them leaks the label.

**The join must be at `TXN_ID` granularity.** Joining customer-level features directly
onto `RAW_TRANSACTIONS` produces ~20 rows per customer that share one feature vector but
carry different labels. That makes the target unlearnable and pins precision to the base
rate. `FRAUD_SCORING_FEATURES` asserts one row per transaction, and the training job fails
loudly if duplicate `TXN_ID`s appear.

## How It Works

**Code-driven promotion:** feature branch → PR (lint+test) → merge to main → deploy-stage (train, validate, register) → human approval → deploy-prod (SPCS, gateway, monitor, tag)

**Data-driven promotion:** weekly cron retrains on fresh data → registers candidate in STAGE → creates GitHub Issue → human reviews → manual promote to PROD

## For Data Science Teams

Swap in your own model logic — the infrastructure stays the same:

1. **Replace the training function** in `source/pipeline/ml_pipeline_dag.py` (the `train_model()` `@remote` function)
2. **Update feature engineering** in `source/features/feature_views.py` and the inlined SQL in `build_feature_eng_remote()`
3. **Update the feature contract** in `source/config.py` (`FEATURE_COLUMNS`) — training, batch inference, the SPCS health check and the endpoint tests all read from it, so there is one place to change
4. **Adjust quality gate thresholds** in `source/config.py` (`MIN_AUC_ROC`, `MIN_PRECISION`, `MIN_RECALL`)

Everything else (CI/CD, versioning, deployment, monitoring) works automatically.

## For ML Engineering Teams

This template provides:

- Standardized ML pipeline structure (Task DAG with configurable compute)
- Automated CI/CD with model approval gates (no model goes to PROD without human review)
- Environment isolation (DEV → STAGE → PROD as separate databases)
- Reproducible deployments (every PROD release tagged, every model version tracked)
- Rollback capability (one-click revert via GitHub Actions)

## Pipeline Flow

### Code-Driven Promotion (push to main)

1. **PR Checks** — Ruff linting and unit tests run automatically on every pull request
2. **Merge to main** — Triggers the deploy workflow
3. **Deploy Task DAG** — Creates/updates Snowflake Tasks using the Python SDK
4. **Execute Pipeline** — Runs three ML Jobs sequentially on the compute pool:
   - Feature Engineering (registers Feature Views from raw data)
   - Model Training (XGBoost with cross-validation, logs to Experiment Tracking)
   - Evaluation (computes metrics, writes to PIPELINE_RESULTS table)
5. **Quality Gate** — Checks AUC-ROC, precision, and recall against configured thresholds. If any threshold is missed, the workflow fails and the model is NOT registered.
6. **Register Model** — Only if the quality gate passes. Auto-increments version (V1 → V2 → V3) in the STAGE Model Registry. Does NOT touch PROD yet.
7. **Batch Inference Validation** — Scores the Feature View table using `model.run()` on the warehouse. Validates predictions are sane (no nulls, probabilities sum to 1.0). Writes results to `BATCH_PREDICTIONS` for monitoring.
8. **Monitor Validation** — Creates a ModelMonitor in STAGE, verifies it reaches ACTIVE state, then drops it (proves PROD setup will succeed)
9. **Human Approval** — Workflow pauses. Reviewer sees metrics in the Job Summary and approves PROD deployment.
10. **PROD Deployment:**
    - **Promote model** — replicates the validated version from STAGE to PROD, sets as DEFAULT
    - Registers Feature Views in PROD
    - **Batch inference** — validates `model.run()` works in PROD
    - **Real-time inference** — deploys SPCS container service (blue/green), shifts Gateway traffic to new version
    - Sets up persistent ModelMonitor (tracks prediction drift daily)
    - Tags release: `prod/V3-20260812-...`

### Data-Driven Promotion (scheduled retrain)

1. **Weekly cron fires** (Monday 6AM PT) — or manually triggered
2. **Same pipeline runs** — Feature Eng → Train → Evaluate on fresh data
3. **STAGE_ONLY mode** — Model is registered in STAGE but NOT promoted to PROD
4. **GitHub Issue created** — "Model Candidate Ready: V4" with metrics table
5. **Human reviews** the issue, decides whether to promote
6. **Manual promote** — Run the deploy workflow with `promote_only=true`, approve PROD deployment

### Rollback

Manual dispatch of the rollback workflow: sets DEFAULT version back, redeploys SPCS service, validates gateway.

## Snowflake Services Used

| Component | Service |
|-----------|---------|
| Pipeline Orchestration | Tasks (Python SDK DAG) |
| Feature Engineering | Feature Store (Dynamic Tables) |
| Model Training | ML Jobs (`@remote` on Compute Pools) |
| Experiment Tracking | ExperimentTracking API |
| Model Versioning | Model Registry |
| Batch Inference | Model Registry `run()` on warehouse |
| Real-Time Serving | SPCS containers + Gateway |
| Model Monitoring | ML Observability (ModelMonitor) |
| CI/CD | GitHub Actions with OIDC (zero secrets) |

## Prerequisites

- Snowflake account with `ACCOUNTADMIN` role (for initial setup; CI uses `MLOPS_DEPLOY_ROLE`)
- Python 3.12+ with [uv](https://docs.astral.sh/uv/) installed
- [Snowflake CLI](https://docs.snowflake.com/en/developer-guide/snowflake-cli/index) (`snow`) installed and configured
- [GitHub CLI](https://cli.github.com/) (`gh`) installed and authenticated
- GitHub repository (public, so Environment protection rules and required reviewers are available)
- **macOS only:** `brew install libomp` — xgboost cannot load without it, so notebook 03 and the local quality-gate script fail with `Library not loaded: @rpath/libomp.dylib`

### Compute pool availability

SPCS compute pools must be enabled on the account. The pipeline runs feature
engineering, training and evaluation as ML Jobs on `CPU_X64_M` nodes, and serves
real-time inference from a container on the same family.

## Getting Started

### 1. Clone and Install

```bash
git clone https://github.com/sfc-gh-takmal/snowflake-mlops-techup.git
cd snowflake-mlops-techup
uv sync
```

### 2. Create Snowflake Infrastructure

```bash
bash scripts/setup.sh
```

Creates three environments (`SNOW_MLOPS_DEV`, `SNOW_MLOPS_STAGE`, `SNOW_MLOPS_PROD`) with databases, warehouses, compute pools, and stages.

Run this as `ACCOUNTADMIN`. It also grants write access on everything it creates to
`DEV_ROLE` (default `SYSADMIN`) — the role your `connections.toml` uses for interactive
work and for the notebooks. Without that, `generate_dataset.py` fails with
`Insufficient privileges to operate on schema 'ML'`.

```bash
# if your connection uses a different role
DEV_ROLE=MY_ANALYST_ROLE bash scripts/setup.sh
```

To tear everything down and start over:

```bash
bash scripts/teardown.sh          # drop the 3 databases + serving objects
bash scripts/teardown.sh --all    # also drop pools, warehouses, role, policy db
```

### 3. Generate Sample Data

```bash
uv run python scripts/generate_dataset.py
```

Creates 100K synthetic fraud transactions (~3% fraud rate) in `SNOW_MLOPS_PROD.ML`.

### 4. Deploy and Run the Pipeline (DEV)

```bash
# Deploy the Task DAG
uv run python source/pipeline/ml_pipeline_dag.py --deploy --env dev

# Execute (triggers Feature Eng → Training → Evaluation)
uv run python source/pipeline/ml_pipeline_dag.py --execute --env dev

# Check status
uv run python source/pipeline/ml_pipeline_dag.py --status --env dev
```

### 5. Set Up CI/CD

```bash
bash scripts/setup_cicd.sh
```

This auto-detects the repo from your git remote, resolves the numeric owner/repo IDs
GitHub embeds in OIDC subject claims, then creates:

- `GITHUB_ACTIONS_POLICY` network policy (Snowflake-managed GitHub runner IP rule)
- `MLOPS_CICD.SECURITY.MLOPS_OIDC_POLICY` authentication policy allowing `WORKLOAD_IDENTITY`
- `MLOPS_DEPLOY_ROLE` with least-privilege grants
- `SVC_GITHUB_ACTIONS_STAGE` and `SVC_GITHUB_ACTIONS` service users (OIDC, one subject each)
- The 8 GitHub repo variables, `STAGE`/`PROD` environments, and branch protection

**If CI fails with `Authentication attempt rejected by the current authentication policy`,**
an account-level authentication policy is blocking OIDC before the subject is ever
evaluated. That is what `MLOPS_OIDC_POLICY` is for — a user-level policy overrides the
account default. Verify with:

```bash
snow sql -q "DESCRIBE AUTHENTICATION POLICY MLOPS_CICD.SECURITY.MLOPS_OIDC_POLICY"
```

The policy lives in its own `MLOPS_CICD` database on purpose: a policy attached to a user
blocks `DROP DATABASE` on whatever database holds it, so keeping it out of `SNOW_MLOPS_*`
means teardown never gets wedged.

### 6. Test End-to-End

```bash
# Create a feature branch, make a change, push
git checkout -b feature/my-change
# ... modify model logic, features, or config ...
git add -A && git commit -m "Update model" && git push -u origin feature/my-change

# Create PR → lint+test run → merge → deploy-stage → approve → deploy-prod
gh pr create --title "My model update"
```

## Project Structure

```
snowflake-mlops/
├── .github/workflows/
│   ├── pr-checks.yml              # PR: lint + test
│   ├── deploy.yml                 # Main: train → promote → deploy (with approval)
│   ├── scheduled-retrain.yml      # Cron: retrain → STAGE candidate → notify
│   └── rollback.yml               # Manual: revert to previous version
├── deploy/                        # Promotion strategies (single/multi-account)
├── scripts/
│   ├── setup.sh                   # Infrastructure provisioning
│   ├── teardown.sh                # Drop everything so you can rebuild clean
│   ├── setup_cicd.sh              # OIDC users + auth policy + network policy + repo config
│   ├── generate_dataset.py        # Synthetic data generation
│   ├── wait_for_task.py           # Poll Task DAG completion
│   ├── quality_gate_and_register.py  # Metric validation + model registration
│   ├── run_batch_inference.py     # Batch scoring + validation
│   ├── deploy_prod_service.py     # Blue/green SPCS deployment
│   ├── setup_model_monitor.py     # Model monitoring setup/validation
│   └── notify_candidate.py        # GitHub Issue notification for candidates
├── source/
│   ├── config.py                  # All configuration (single file)
│   ├── snowpark_session.py        # Session helper (SSO + OIDC)
│   ├── features/                  # Feature Store definitions + scoring view
│   ├── pipeline/
│   │   └── ml_pipeline_dag.py     # Task DAG + @remote ML Job functions
│   ├── serving/
│   │   ├── batch_inference.py     # Batch inference utilities
│   │   └── samples.py             # Model input samples drawn from real data
│   └── monitoring/
│       └── model_monitor.py       # Monitor create/status/suspend/resume
├── tests/                         # Unit + integration tests
├── notebooks/                     # Educational Jupyter notebooks (01-05)
└── docs/
    └── docs.html                  # Detailed architecture documentation
```

## Configuration

Everything is configurable from `source/config.py`:

```python
# Compute mode per pipeline step
"feature_engineering_compute": "spcs",  # or "warehouse"
"training_compute": "spcs",             # "warehouse" is rejected at deploy time:
                                        # warehouse UDFs cannot provide xgboost/sklearn
"evaluation_compute": "spcs",

# Deployment toggles
"deploy_batch_inference": "true",
"deploy_realtime_service": "true",
"enable_model_monitor": "true",

# Task timeout (wired into both the DAG and each DAGTask)
"task_timeout_ms": "7200000",  # 2 hours; Snowflake's own default is only 1h

# Feature View refresh
"customer_features_refresh": "1 hour",

# Scheduled retraining
"schedule": "USING CRON 0 6 * * MON America/Los_Angeles",

# Quality gate thresholds -- SINGLE SOURCE OF TRUTH.
# PIPELINE_CONFIG derives from these, so there is no second set of numbers that
# can silently disagree with what CI actually enforces.
MIN_AUC_ROC = 0.80
MIN_PRECISION = 0.12   # base fraud rate is 3%, so this is a 4x lift
MIN_RECALL = 0.50

# The model's feature contract -- also single source of truth
FEATURE_COLUMNS = TRANSACTION_FEATURE_COLUMNS + CUSTOMER_FEATURE_COLUMNS
```

## CI/CD Workflows

| Trigger | What Happens | Human Approval |
|---------|-------------|----------------|
| PR to `main` | Lint + unit tests | Merge requires passing checks |
| Push to `main` | Train → quality gate → register → batch inference → PROD deploy | Required for PROD |
| Weekly cron | Retrain on fresh data → register candidate → GitHub Issue | Manual promote via `promote_only` |
| Manual dispatch | Promote existing candidate or full retrain | Required for PROD |
| Rollback dispatch | Revert PROD to previous version | Immediate |

## Extending This Template

**Add a new pipeline step:**
1. Add `"new_step_compute": "spcs"` to `PIPELINE_CONFIG`
2. Create a `build_new_step_remote(cfg)` function in `ml_pipeline_dag.py`
3. Add `DAGTask("NEW_STEP", definition=func)` and wire dependencies

**Switch to a different model:**
1. Replace the training logic in `build_train_model_remote()`
2. Update feature columns and Feature View SQL
3. Adjust quality gate thresholds

**Add multi-account support:**
1. Implement `deploy/strategies/multi_account.py`
2. Set `TOPOLOGY=multi-account` in GitHub variables

## Documentation

Open `docs/docs.html` in a browser for a detailed Level 300 walkthrough covering the Task DAG, ML Jobs, Feature Store, Model Registry, inference deployment, CI/CD pipeline, monitoring, and lessons learned.

## Operational notes

Things that are easy to get wrong and cost a CI run each to discover.

### Handing interactively-built objects to the CI role

`GRANT ALL` does **not** include `OWNERSHIP`, and the pipeline uses
`CREATE OR REPLACE` in several places. If you build the demo interactively first and
then let CI take over, `setup_cicd.sh` transfers what it can:

| Object | Needed because | Note |
|--------|----------------|------|
| Dynamic tables | Feature Store replaces feature views on every register | `OWNERSHIP` |
| Views | `FRAUD_SCORING_FEATURES` is `CREATE OR REPLACE VIEW` | `OWNERSHIP` |
| Tables | `BATCH_PREDICTIONS` written with `mode("overwrite")` | `OWNERSHIP` |
| Tasks | DAG deployed with `CreateMode.or_replace` | must be **suspended** first |
| Tags | Feature Store re-stamps metadata tags each register | `APPLY`, **no bulk form** |
| Experiments | `set_experiment()` on an existing experiment | **no bulk form** |

Two object types **cannot** be transferred: `TRANSFER OWNERSHIP ON SERVICE` is
unsupported, and the same applies in practice to model monitors. A service created by
another role can never be dropped by CI, which permanently breaks blue/green cleanup.
`scripts/teardown.sh` drops them so CI recreates and owns them.

### OIDC tokens expire mid-job

`deploy-stage` takes ~10 minutes and `deploy-prod` ~12. The OIDC JWT minted when
`snowflake-actions` runs does not survive that, and re-running the action mid-job does
**not** reliably refresh it. `create_snowpark_session()` therefore mints a fresh token
from `ACTIONS_ID_TOKEN_REQUEST_URL` at the moment it connects, so elapsed job time is
irrelevant. This needs `permissions: id-token: write` on the job (already set).

`scripts/verify_oidc_minting.py` exercises all three paths — no-op outside Actions,
fresh token overriding a stale one, and graceful fallback on endpoint failure.

### The approval gate

GitHub does not allow approving your own pull request, so `REQUIRED_REVIEWS` defaults
to `0` and PR review is opt-in. Direct pushes to `main` are still blocked by required
status checks plus `enforce_admins`.

The human-approval story lives on the **PROD environment reviewer gate**, which *can*
be self-approved — so a single operator can still demonstrate it end to end. Set
`REQUIRED_REVIEWS=1` on a repo with more than one maintainer.

Note that `required_pull_request_reviews` blocks merging whenever the block is
*present*, even with a count of `0`; it has to be `null`.


## License

MIT
