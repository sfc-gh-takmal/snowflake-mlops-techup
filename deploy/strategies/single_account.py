"""Single-account model promotion strategy.

Promotes a model from STAGE to PROD by cross-database replication
within the same Snowflake account.

SQL pattern:
    ALTER MODEL PROD_DB.SCHEMA.MODEL ADD VERSION V{n}
      FROM MODEL STAGE_DB.SCHEMA.MODEL VERSION V{n}
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "source"))
from snowpark_session import create_snowpark_session

STAGE_DATABASE = "SNOW_MLOPS_STAGE"
STAGE_SCHEMA = "ML"
PROD_DATABASE = "SNOW_MLOPS_PROD"
PROD_SCHEMA = "ML"
MODEL_NAME = "MLOPS_FRAUD_DETECTOR"


def promote(version: str, session=None):
    """Replicate a model version from STAGE to PROD (cross-database copy)."""
    close_session = False
    if session is None:
        session = create_snowpark_session()
        close_session = True

    print(f"[single-account] Promoting {MODEL_NAME}/{version}: {STAGE_DATABASE} -> {PROD_DATABASE}")

    # Check if model exists in PROD
    existing = session.sql(f"SHOW MODELS LIKE '{MODEL_NAME}' IN {PROD_DATABASE}.{PROD_SCHEMA}").collect()

    if existing:
        # Add version to existing model
        try:
            session.sql(f"""
                ALTER MODEL {PROD_DATABASE}.{PROD_SCHEMA}.{MODEL_NAME}
                  ADD VERSION {version}
                  FROM MODEL {STAGE_DATABASE}.{STAGE_SCHEMA}.{MODEL_NAME}
                  VERSION {version}
            """).collect()
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"  Version {version} already exists in PROD, dropping and re-adding...")
                session.sql(f"ALTER MODEL {PROD_DATABASE}.{PROD_SCHEMA}.{MODEL_NAME} DROP VERSION {version}").collect()
                session.sql(f"""
                    ALTER MODEL {PROD_DATABASE}.{PROD_SCHEMA}.{MODEL_NAME}
                      ADD VERSION {version}
                      FROM MODEL {STAGE_DATABASE}.{STAGE_SCHEMA}.{MODEL_NAME}
                      VERSION {version}
                """).collect()
            else:
                raise
    else:
        # Create model with first version
        session.sql(f"""
            CREATE MODEL {PROD_DATABASE}.{PROD_SCHEMA}.{MODEL_NAME}
              WITH VERSION {version}
              FROM MODEL {STAGE_DATABASE}.{STAGE_SCHEMA}.{MODEL_NAME}
              VERSION {version}
        """).collect()

    # Set as default version in PROD
    session.sql(f"""
        ALTER MODEL {PROD_DATABASE}.{PROD_SCHEMA}.{MODEL_NAME}
          SET DEFAULT_VERSION = {version}
    """).collect()

    # Verify
    result = session.sql(f"SHOW MODELS LIKE '{MODEL_NAME}' IN {PROD_DATABASE}.{PROD_SCHEMA}").collect()
    if result:
        print(f"  Promoted successfully! Default version: {version}")
        print(f"  All PROD versions: {result[0]['versions']}")
    else:
        raise RuntimeError("Promotion failed - model not found in PROD after replication")

    if close_session:
        session.close()
