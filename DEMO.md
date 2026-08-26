# Demo Runbook

A modular demo of the Snowflake-native MLOps pipeline. Every segment is
self-contained, so you can run three of them in ten minutes or all of them in
thirty, in any order.

**The design constraint:** a full CI run takes ~22 minutes (10 for
`deploy-stage`, 12 for `deploy-prod`). You cannot watch that on stage. So this
runbook demos **artifacts that already exist** plus a handful of genuinely
instant live commands. The pipeline is shown as a *completed run* you click
through, not something the room waits for.

Every command below was executed against the live account and the outputs are
the real ones. If what you see differs materially, something has changed —
check the fallback note.

---

## Pre-flight (do this 15 minutes before)

```bash
# 1. Seed the quality-gate fixtures
cd /path/to/snowflake-mlops-techup
uv run python scripts/seed_demo_metrics.py

# 2. Confirm the model, service and gateway are healthy
snow sql -c SNOW_MLOPS -q "SHOW VERSIONS IN MODEL SNOW_MLOPS_PROD.ML.MLOPS_FRAUD_DETECTOR"

# 3. Get the CURRENT gateway hostname (it is not stable across recreation)
snow sql -c SNOW_MLOPS -q "DESC GATEWAY SNOW_MLOPS_PROD.ML.FRAUD_DETECTOR_GATEWAY"
```

Checklist:

- [ ] `/tmp/demo_good.json` and `/tmp/demo_degraded.json` exist
- [ ] Model `MLOPS_FRAUD_DETECTOR` shows V1, V3, V4 with **V4** as `DEFAULT`
- [ ] Exactly one service `MLOPS_FRAUD_DETECTOR_SERVICE_V4` is `RUNNING`
- [ ] Streamlit app loads: **Projects > Streamlit > MLOPS_CONTROL_PLANE**
- [ ] GitHub run [`32916612949`](https://github.com/sfc-gh-takmal/snowflake-mlops-techup/actions/runs/32916612949) open in a tab
- [ ] Terminal font size raised; `cd` into the repo

> **Warm the pools only if you intend to run something live.** `SNOW_MLOPS_DEV_POOL`
> and `SNOW_MLOPS_STAGE_POOL` sit `SUSPENDED` and cold-start takes 2-3 minutes.
> `SNOW_MLOPS_PROD_POOL` is `ACTIVE` and serves everything in this runbook.

---

## Timing plan

| Segment | Time | 10-min | 15-min | 30-min |
|---|---|:--:|:--:|:--:|
| 1. Framing | 2 min | ✓ | ✓ | ✓ |
| 2. Control plane (Streamlit) | 4 min | ✓ | ✓ | ✓ |
| 3. Quality gate blocking a bad model | 3 min | ✓ | ✓ | ✓ |
| 4. SQL-native inference | 2 min | — | ✓ | ✓ |
| 5. Feature store as Dynamic Tables | 3 min | — | ✓ | ✓ |
| 6. The completed CI run + approval | 5 min | — | — | ✓ |
| 7. Blue/green serving | 3 min | — | — | ✓ |
| 8. Monitoring | 3 min | — | — | ✓ |
| 9. What it took (internal only) | 5 min | — | — | ✓ |

The 10-minute cut is segments **1 → 2 → 3**. That is the whole story in
miniature: here is the system, here is it working, here is it refusing to ship
a bad model.

---

## 1. Framing (2 min, no commands)

Five problems every ML team has, and where each is solved here:

| Problem | Solved by |
|---|---|
| Features computed differently in training vs serving | Feature Store — one definition, both paths |
| "Which model is in production?" | Model Registry with versions and aliases |
| A bad model reaching production | Quality gate in CI + human approval |
| Deploying a new version means downtime | Gateway traffic split (blue/green) |
| No idea if the model still works | Model Monitor |

The point to land: **no external orchestrator, no data leaving Snowflake, and
no long-lived secrets in CI** (GitHub OIDC federation, keyless).

---

## 2. Control plane — the Streamlit app (4 min)

**Open:** Snowsight → Projects → Streamlit → `MLOPS_CONTROL_PLANE`

This is the safest opener: one screen, live data, nothing to type.

Walk the tabs in this order:

1. **Overview** — serving version `V4`, 3 registered versions, 1 live inference
   service, 100,000 transactions scored. Held-out metrics beside the promotion
   gate thresholds, all `PASS`.
2. **Registry & Features** — version history with aliases and attached service;
   two feature views with `1 hour` target lag.
3. **Predictions** — risk distribution. 62% of transactions land in the minimal
   band, 2.31% in critical.
4. **Threshold Explorer** — drag the slider. This is the interactive moment.
5. **Serving & Monitoring** — services, gateway traffic split, monitor state.

**What it proves:** the whole lifecycle is queryable as data. Nothing here is a
screenshot.

**Say this on the PR-AUC number:** 0.41 against a ~3% fraud base rate is ~13x
better than random. AUC-ROC of 0.895 sounds better but is the less honest
metric on imbalanced data — leading with PR-AUC builds credibility with the
technical half of the room.

**On the Threshold Explorer, read the banner out loud.** Those metrics are
**in-sample** — scored over the same rows the model trained on — which is why
recall reads 0.91 there but 0.665 on the Overview tab. Use the slider to show
the *shape* of the precision/recall trade-off, not to claim performance. Getting
ahead of this earns trust; being caught on it loses the room.

**Fallback:** if the app is slow to wake (container cold start, ~30s), hit
**Refresh** once. If it still misbehaves, every panel has a SQL equivalent in
the segments below — skip to those.

---

## 3. Quality gate blocking a bad model (3 min) — **the strongest segment**

Governance claims are cheap. A gate that visibly refuses is not. This runs
**offline in under a second** because the gate exits before it ever opens a
Snowflake session.

**Passing model:**

```bash
METRICS_FILE=/tmp/demo_good.json uv run python scripts/quality_gate_and_register.py
```

```
============================================================
QUALITY GATE + MODEL REGISTRATION
============================================================

[1/3] Quality gate check...
  PASSED — all metrics meet thresholds

[2/3] Registering model...
```

> Interrupt with `Ctrl-C` once you see `PASSED`. Letting it continue registers a
> real model version in PROD, which you do not want mid-demo.

**Regressed model:**

```bash
METRICS_FILE=/tmp/demo_degraded.json uv run python scripts/quality_gate_and_register.py
echo "exit code: $?"
```

```
[1/3] Quality gate check...
  FAILED:
    - AUC-ROC 0.6200 < 0.8
    - Precision 0.0400 < 0.12
    - Recall 0.3100 < 0.5

Model will NOT be registered. Fix and re-run.
exit code: 1
```

**What it proves:** three things worth calling out. The gate evaluates *all*
metrics rather than short-circuiting on the first failure. It exits non-zero, so
CI stops and `deploy-prod` never starts. And the model is never registered —
there is no bad artifact left behind to clean up.

**Say this:** the thresholds live in `source/config.py` as the single source of
truth. Changing them is a pull request, which means changing your definition of
"good enough" is reviewed like code.

**Fallback:** if the fixtures are missing, `uv run python scripts/seed_demo_metrics.py`
regenerates both in a second.

---

## 4. SQL-native inference (2 min)

No endpoint, no SDK, no data movement — the model is a SQL function.

```sql
SELECT
  f.TXN_ID,
  ROUND(f.AMOUNT, 2)              AS amount,
  f.IS_LATE_NIGHT                 AS late_night,
  ROUND(f.MERCHANT_RISK_SCORE, 2) AS merchant_risk,
  ROUND(SNOW_MLOPS_PROD.ML.MLOPS_FRAUD_DETECTOR!PREDICT_PROBA(
      f.AMOUNT, f.AMOUNT_TO_AVG_RATIO, f.IS_HIGH_RISK_MERCHANT, f.MERCHANT_RISK_SCORE,
      f.HOUR_OF_DAY, f.IS_WEEKEND, f.IS_LATE_NIGHT,
      f.TOTAL_TXN_COUNT, f.AVG_TXN_AMOUNT, f.MAX_TXN_AMOUNT, f.STDDEV_TXN_AMOUNT,
      f.UNIQUE_MERCHANTS, f.ACTIVE_DAYS, f.LATE_NIGHT_TXN_RATIO,
      f.CREDIT_SCORE, f.ACCOUNT_AGE_DAYS, f.ANNUAL_INCOME
  ):output_feature_1::FLOAT, 4)   AS fraud_probability
FROM SNOW_MLOPS_PROD.ML.FRAUD_SCORING_FEATURES f
ORDER BY fraud_probability DESC
LIMIT 5;
```

Returns in ~2 seconds:

| TXN_ID | AMOUNT | LATE_NIGHT | MERCHANT_RISK | FRAUD_PROBABILITY |
|---|---|---|---|---|
| TXN_0000486 | 735.36 | 1 | 0.78 | 0.9995 |
| TXN_0062778 | 1855.42 | 1 | 0.67 | 0.9994 |
| TXN_0043892 | 3109.35 | 1 | 0.87 | 0.9993 |
| TXN_0052746 | 998.40 | 1 | 0.85 | 0.9993 |
| TXN_0024311 | 3453.12 | 1 | 0.84 | 0.9992 |

**What it proves:** any analyst with SQL access can score data. The model is
governed like any other Snowflake object.

**Point at the columns:** every top-5 transaction is late-night with an
elevated merchant risk score. The model learned something a human would
recognise — it is not a black box producing arbitrary numbers.

**Gotchas:** the 17 feature arguments are **positional** — order matters, and
it is defined by `FEATURE_COLUMNS` in `source/config.py`. The output column
`output_feature_1` is lower-case, so it must stay quoted in any other SQL you
write against it.

---

## 5. Feature store as Dynamic Tables (3 min)

```sql
SHOW DYNAMIC TABLES IN SCHEMA SNOW_MLOPS_PROD.ML;
```

| Feature view | Rows | Target lag | Refresh mode | State |
|---|---|---|---|---|
| `CUSTOMER_RISK_FEATURES$V1` | 5,000 | 1 hour | INCREMENTAL | ACTIVE |
| `TRANSACTION_CONTEXT_FEATURES$V1` | 100,000 | 1 hour | FULL | ACTIVE |

**What it proves:** feature freshness is *declared*, not scheduled. You state a
target lag and Snowflake maintains it. There is no Airflow DAG to babysit.

**Two honest details worth volunteering:**

- `CUSTOMER_RISK_FEATURES` refreshes `INCREMENTAL`; `TRANSACTION_CONTEXT_FEATURES`
  fell back to `FULL` because its query is too complex to diff incrementally.
  Snowflake tells you this in `refresh_mode_reason` rather than silently
  degrading. That transparency is the feature.
- The customer feature view also computes `HISTORICAL_FRAUD_COUNT` and
  `HISTORICAL_FRAUD_RATE`, and the model **deliberately does not consume them**.
  They are computed from the label, so training on them would be target leakage.
  This is worth 30 seconds — it is the kind of bug that produces an AUC of 0.99
  in development and a useless model in production.

---

## 6. The completed CI run and the approval gate (5 min)

Open run [`32916612949`](https://github.com/sfc-gh-takmal/snowflake-mlops-techup/actions/runs/32916612949).

Walk it in this order:

1. **The trigger** — a merged PR. `lint` and `test` are required status checks,
   so an unmergeable PR never reaches deploy.
2. **`deploy-stage`** — expand the job summary. The metrics table is rendered by
   the pipeline itself, not pasted in.
3. **The wait** — the run *paused* here for PROD approval.
4. **The approval record** — who approved, and when. This is the artifact
   auditors ask for.
5. **`deploy-prod`** — feature views registered, model promoted, service built,
   gateway traffic shifted, old service dropped.
6. **The release tag** — `prod/V4-20260826-011113`, tying a git SHA to a model
   version.

**What it proves:** the promotion path is a reviewed, auditable, repeatable
workflow. And there are **no Snowflake credentials in GitHub** — authentication
is OIDC workload identity federation, so there is no secret to rotate or leak.

**Say this on timing:** ~10 minutes to STAGE, ~12 to PROD, most of it building
the inference container. That is why we are looking at a finished run rather
than starting one.

---

## 7. Blue/green serving (3 min)

```sql
SHOW SERVICES IN SCHEMA SNOW_MLOPS_PROD.ML;
```

| Service | Status | Owner |
|---|---|---|
| `MLOPS_FRAUD_DETECTOR_SERVICE_V4` | RUNNING | `MLOPS_DEPLOY_ROLE` |
| `MODEL_BUILD_5D6E919A` | DONE | (build job for V1) |
| `MODEL_BUILD_6B988D3B` | DONE | (build job for V4) |
| `MODEL_BUILD_B7F3A398` | DONE | (build job for V3) |

Exactly one inference service is running. The `MODEL_BUILD_*` rows are finished
container builds — useful history, not endpoints. **Note what is absent:**
there is no `..._SERVICE_V3`. CI created V4, shifted traffic, then dropped V3.

```sql
DESC GATEWAY SNOW_MLOPS_PROD.ML.FRAUD_DETECTOR_GATEWAY;
```

```yaml
spec:
  type: traffic_split
  split_type: custom
  targets:
    - type: endpoint
      value: SNOW_MLOPS_PROD.ML.MLOPS_FRAUD_DETECTOR_SERVICE_V4!inference
      weight: 100
```

**What it proves:** clients address the gateway, never a versioned service.
Shipping a new version is a weight change, so there is no client-visible
downtime and no coordinated cutover.

**The natural follow-up — "can you canary?"** Yes: set two targets with weights
90/10. This deployment ships at 100% because the quality gate and the approval
step already gate the risk.

**Fallback:** the gateway hostname is regenerated if the gateway is recreated,
so read it from `DESC GATEWAY` rather than from a slide.

---

## 8. Monitoring (3 min)

```sql
SHOW MODEL MONITORS IN SCHEMA SNOW_MLOPS_PROD.ML;
```

`FRAUD_DETECTOR_MONITOR` — `ACTIVE`, refresh `1 day`, aggregation window
`7 days`, bound to `MLOPS_FRAUD_DETECTOR` version `V4`.

```sql
SELECT * FROM TABLE(MODEL_MONITOR_STAT_METRIC(
  'SNOW_MLOPS_PROD.ML.FRAUD_DETECTOR_MONITOR',
  'COUNT', '"output_feature_1"', '1 DAY'));
```

| EVENT_TIMESTAMP | METRIC_VALUE | METRIC_NAME | COLUMN_NAME |
|---|---|---|---|
| 2026-08-20 00:00:00 | 100000 | COUNT | output_feature_1 |

**What it proves:** monitoring is a first-class object bound to a specific model
version, refreshing on a schedule, queryable in SQL.

**Be precise about scope.** This monitor tracks prediction **volume and
distribution statistics**. Its `baseline` is `NOT_SET`, so it does **not**
compute drift. Say that plainly — adding a baseline is a config change, and
claiming drift you cannot show is the fastest way to lose a technical audience.

**Gotchas if you type this live:** the third argument is a *column name*, not
granularity, and `output_feature_1` is lower-case so it needs the nested quoting
`'"output_feature_1"'`. Granularity is the fourth argument and must be a
literal.

---

## 9. What it took (5 min, internal audiences only)

Skip this for customers. For SEs and ML engineers it is the most valuable part,
because it reframes the demo from "look, it works" to "here is what production
actually requires."

Building this surfaced 17 defects. The three worth telling:

**The join fan-out.** Training joined per-transaction labels to customer-level
features, fanning 5,000 customers into 100,000 rows — and 2,271 customers
carried *both* fraud and non-fraud labels. The model could not separate them, so
precision was pinned near the base rate. Fixed by scoring at transaction
granularity with a fan-out assertion in the pipeline.

**An AUC of exactly 1.000.** After fixing the join, the model became perfect —
always a bug, never a triumph. The synthetic generator drew fraud at hours
`{0-5, 23}` and legitimate transactions at `6-22`. Disjoint support means hour
alone was a perfect classifier. Fixed with overlapping hour distributions.

**Target leakage.** The customer feature view computes `HISTORICAL_FRAUD_RATE`
from the label column. Including it would have looked excellent in development
and failed in production.

**The lesson:** the platform makes the plumbing easy, which means the remaining
hard part is ML correctness. Fast infrastructure lets you find these bugs
sooner — it does not find them for you.

---

## Landmines

| Risk | Mitigation |
|---|---|
| `rollback.yml` and `scheduled-retrain.yml` have **never been run end-to-end** | Do not demo them live. Input validation was verified in isolation; the workflows were not. |
| Cold compute pools (DEV/STAGE `SUSPENDED`) | Everything in this runbook uses PROD, which is `ACTIVE`. Resume the others only if going live. |
| Letting the passing gate finish | `Ctrl-C` after `PASSED` — it registers a real PROD version otherwise. |
| Gateway hostname changes on recreation | Read it from `DESC GATEWAY` at pre-flight. |
| Notebooks are local Jupyter and need `libomp` | Run `01`→`05` in order on one machine; `03` hands off to `04` via `/tmp`. |
| Re-triggering deploy repeatedly | Each run registers a new version and rebuilds the container (~6 min). |
| README and `docs/docs.html` still mention drift tracking | Known overclaim. Steer monitoring to volume/statistics, per segment 8. |

---

## Questions you will get

**"Does this work with dbt / Airflow / SageMaker?"**
Yes — the pipeline is Snowflake Tasks and ML Jobs, callable from anything that
can issue SQL. Nothing here forbids an external orchestrator; the point is you
do not *need* one.

**"How do you roll back?"**
Move the `DEFAULT` alias to a prior version and shift gateway traffic. V1 and V3
are still registered. There is a `rollback.yml` workflow — say honestly that it
has not been exercised end-to-end.

**"What does this cost?"**
Warehouses for training and features; a compute pool for the inference service
and container builds. The pool is the item to size deliberately —
`SNOW_MLOPS_PROD_POOL` is `CPU_X64_M`, 1-3 nodes.

**"How do you detect drift?"**
Not currently. The monitor tracks volume and distribution statistics; drift
comparison needs a registered baseline, which this deployment does not define.
It is a config change, not an architecture change.

**"Why XGBoost and not deep learning?"**
Tabular fraud detection with 17 features. Gradient boosting is the right tool;
the pipeline is model-agnostic.

---

## Reference

| Thing | Value |
|---|---|
| Repo | `sfc-gh-takmal/snowflake-mlops-techup` |
| Green run | [`32916612949`](https://github.com/sfc-gh-takmal/snowflake-mlops-techup/actions/runs/32916612949) |
| Connection | `SNOW_MLOPS` |
| Databases | `SNOW_MLOPS_DEV` / `_STAGE` / `_PROD`, schema `ML` |
| Model | `SNOW_MLOPS_PROD.ML.MLOPS_FRAUD_DETECTOR` (V1, V3, **V4 default**) |
| Service | `MLOPS_FRAUD_DETECTOR_SERVICE_V4` |
| Gateway | `FRAUD_DETECTOR_GATEWAY` |
| Monitor | `FRAUD_DETECTOR_MONITOR` |
| Streamlit app | `SNOW_MLOPS_PROD.ML.MLOPS_CONTROL_PLANE` |
| Held-out metrics | AUC 0.8952 · PR-AUC 0.4081 · P 0.1964 · R 0.6650 · F1 0.3032 |
| Gate thresholds | AUC ≥ 0.80 · Precision ≥ 0.12 · Recall ≥ 0.50 |
