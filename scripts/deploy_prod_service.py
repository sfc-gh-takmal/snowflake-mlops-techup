"""Blue/Green PROD deployment with Snowflake Gateway traffic routing.

This script is called by the deploy-prod.yml GitHub Actions workflow.
It deploys a NEW service for the latest model version, validates it,
then shifts 100% gateway traffic to the new service and cleans up the old one.

Flow:
  1. Get latest model version from registry
  2. Create versioned service (e.g. MLOPS_FRAUD_DETECTOR_SERVICE_V3)
  3. Wait for READY state
  4. Health check the new service
  5. ALTER GATEWAY to route 100% traffic to new service
  6. Drop old service
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "source"))
from snowpark_session import create_snowpark_session

PROD_DATABASE = "SNOW_MLOPS_PROD"
PROD_SCHEMA = "ML"
PROD_WAREHOUSE = "SNOW_MLOPS_PROD_WH"
PROD_COMPUTE_POOL = "SNOW_MLOPS_PROD_POOL"
MODEL_NAME = "MLOPS_FRAUD_DETECTOR"
GATEWAY_NAME = "FRAUD_DETECTOR_GATEWAY"
SERVICE_PREFIX = "MLOPS_FRAUD_DETECTOR_SERVICE"

READY_TIMEOUT_SECONDS = 600  # 10 minutes
READY_POLL_INTERVAL = 15


def get_current_gateway_target(session):
    """Get the current service target from the gateway spec. Returns None if gateway doesn't exist."""
    try:
        rows = session.sql(f"DESC GATEWAY {PROD_DATABASE}.{PROD_SCHEMA}.{GATEWAY_NAME}").collect()
    except Exception:
        return None
    if not rows:
        return None
    spec = rows[0]["spec"]
    for line in spec.split("\n"):
        line = line.strip()
        if line.startswith("value:"):
            full_value = line.split("value:")[1].strip()
            service_fqn = full_value.split("!")[0].strip()
            return service_fqn.split(".")[-1]
    return None


def ensure_gateway_exists(session, service_name):
    """Create the gateway if it doesn't exist, pointing to the given service."""
    try:
        session.sql(f"DESC GATEWAY {PROD_DATABASE}.{PROD_SCHEMA}.{GATEWAY_NAME}").collect()
        return  # Already exists
    except Exception:
        pass

    print(f"  Creating gateway: {GATEWAY_NAME}")
    fqn = f"{PROD_DATABASE}.{PROD_SCHEMA}.{service_name}!inference"
    session.sql(f"""
        CREATE GATEWAY {PROD_DATABASE}.{PROD_SCHEMA}.{GATEWAY_NAME}
        FROM SPECIFICATION $$
          spec:
            type: traffic_split
            split_type: custom
            targets:
              - type: endpoint
                value: {fqn}
                weight: 100
        $$
    """).collect()
    print("  Gateway created.")


def wait_for_service_ready(session, service_name, timeout=READY_TIMEOUT_SECONDS):
    """Poll service status until READY or timeout."""
    fqn = f"{PROD_DATABASE}.{PROD_SCHEMA}.{service_name}"
    start = time.time()
    while time.time() - start < timeout:
        try:
            status_json = session.sql(f"SELECT SYSTEM$GET_SERVICE_STATUS('{fqn}')").collect()[0][0]
            statuses = json.loads(status_json)
            ready = [s for s in statuses if s["status"] == "READY"]
            if ready:
                return True
            print(f"  Status: {statuses[0]['status']} (elapsed: {int(time.time() - start)}s)")
        except Exception as e:
            print(f"  Waiting for service to appear... ({e})")
        time.sleep(READY_POLL_INTERVAL)
    return False


def health_check(session, service_name, model_version_name):
    """Run a test prediction against the new service to validate it works."""
    from serving.samples import get_sample
    from snowflake.ml.registry import Registry

    reg = Registry(session=session, database_name=PROD_DATABASE, schema_name=PROD_SCHEMA)
    model = reg.get_model(MODEL_NAME)
    mv = model.version(model_version_name)

    # Draw a real row from the scoring view so columns and dtypes match the model
    # signature exactly, rather than hand-maintaining a literal dict here.
    sample = get_sample(session, PROD_DATABASE, PROD_SCHEMA, n=1)

    result = mv.run(sample, function_name="predict_proba", service_name=service_name)
    # Validate result structure
    assert "output_feature_0" in result.columns, f"Missing output_feature_0 in {result.columns.tolist()}"
    assert "output_feature_1" in result.columns, f"Missing output_feature_1 in {result.columns.tolist()}"
    prob_sum = result["output_feature_0"].iloc[0] + result["output_feature_1"].iloc[0]
    assert abs(prob_sum - 1.0) < 0.01, f"Probabilities sum to {prob_sum}, not 1.0"
    return result


def shift_gateway_traffic(session, new_service_name):
    """ALTER GATEWAY to route 100% traffic to the new service."""
    fqn = f"{PROD_DATABASE}.{PROD_SCHEMA}.{new_service_name}!inference"
    session.sql(f"""
        ALTER GATEWAY {PROD_DATABASE}.{PROD_SCHEMA}.{GATEWAY_NAME}
        FROM SPECIFICATION $$
          spec:
            type: traffic_split
            split_type: custom
            targets:
              - type: endpoint
                value: {fqn}
                weight: 100
        $$
    """).collect()


def cleanup_old_service(session, old_service_name, new_service_name):
    """Drop the old service after traffic has been shifted."""
    if old_service_name and old_service_name != new_service_name:
        print(f"  Dropping old service: {old_service_name}")
        session.sql(f"DROP SERVICE IF EXISTS {PROD_DATABASE}.{PROD_SCHEMA}.{old_service_name}").collect()


def main():
    print("=" * 60)
    print("PROD DEPLOYMENT: Blue/Green with Gateway")
    print("=" * 60)

    session = create_snowpark_session()
    session.sql(f"USE WAREHOUSE {PROD_WAREHOUSE}").collect()

    # Step 1: Get the latest model version
    print("\n[1/6] Getting latest model version...")
    models = session.sql(f"SHOW MODELS LIKE '{MODEL_NAME}' IN {PROD_DATABASE}.{PROD_SCHEMA}").collect()
    if not models:
        raise RuntimeError(f"Model {MODEL_NAME} not found in {PROD_DATABASE}.{PROD_SCHEMA}")

    default_version = models[0]["default_version_name"]
    print(f"  Model: {MODEL_NAME}, Default version: {default_version}")

    # Step 2: Determine service names
    print("\n[2/6] Determining service names...")
    new_service_name = f"{SERVICE_PREFIX}_{default_version}"
    old_service_name = get_current_gateway_target(session)
    print(f"  New service: {new_service_name}")
    print(f"  Old service: {old_service_name or 'none'}")

    if old_service_name == new_service_name:
        print(f"\n  Service {new_service_name} is already the active gateway target.")
        print("  Nothing to deploy. Exiting.")
        session.close()
        return

    # Step 3: Create the new versioned service
    print(f"\n[3/6] Creating service: {new_service_name}...")
    from snowflake.ml.registry import Registry

    reg = Registry(session=session, database_name=PROD_DATABASE, schema_name=PROD_SCHEMA)
    model = reg.get_model(MODEL_NAME)
    mv = model.version(default_version)

    try:
        mv.create_service(
            service_name=new_service_name,
            service_compute_pool=PROD_COMPUTE_POOL,
            image_build_compute_pool=PROD_COMPUTE_POOL,
            ingress_enabled=True,
            max_instances=2,
            gpu_requests=None,
            autocapture=True,
        )
        print("  Service creation initiated.")
    except Exception as e:
        if "already exists" in str(e).lower():
            print("  Service already exists - will verify readiness.")
        else:
            raise

    # Step 4: Wait for READY
    print("\n[4/6] Waiting for service to become READY...")
    ready = wait_for_service_ready(session, new_service_name)
    if not ready:
        raise RuntimeError(f"Service {new_service_name} did not become READY within {READY_TIMEOUT_SECONDS}s")
    print("  Service is READY!")

    # Step 5: Health check
    print(f"\n[5/6] Running health check against {new_service_name}...")
    result = health_check(session, new_service_name, default_version)
    fraud_prob = result["output_feature_1"].iloc[0]
    print(f"  Health check PASSED (fraud_prob={fraud_prob:.4f})")

    # Step 6: Shift gateway traffic + cleanup
    print(f"\n[6/6] Shifting gateway traffic to {new_service_name}...")
    ensure_gateway_exists(session, new_service_name)
    shift_gateway_traffic(session, new_service_name)
    print("  Gateway updated! 100% traffic now routes to new service.")

    # Cleanup old service
    cleanup_old_service(session, old_service_name, new_service_name)

    # Show gateway endpoint. DESC returns a human-readable status string rather than
    # a host while endpoints are still provisioning, so don't blindly prefix https://.
    gw = session.sql(f"DESC GATEWAY {PROD_DATABASE}.{PROD_SCHEMA}.{GATEWAY_NAME}").collect()
    gateway_url = (gw[0]["ingress_url"] or "") if gw else ""
    if gateway_url and " " not in gateway_url:
        print(f"\n  Gateway URL: https://{gateway_url}")
    else:
        print(f"\n  Gateway endpoint not ready yet: {gateway_url or 'provisioning...'}")
        print(f"  Check with: DESC GATEWAY {PROD_DATABASE}.{PROD_SCHEMA}.{GATEWAY_NAME}")

    print("\n" + "=" * 60)
    print("PROD DEPLOYMENT COMPLETE (Blue/Green)")
    print(f"  Model: {MODEL_NAME} ({default_version})")
    print(f"  Service: {new_service_name}")
    print(f"  Gateway: {GATEWAY_NAME}")
    print("=" * 60)
    session.close()


if __name__ == "__main__":
    main()
