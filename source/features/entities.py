"""Register Feature Store entities for fraud detection."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATABASE, SCHEMA, WAREHOUSE
from snowflake.ml.feature_store import CreationMode, Entity, FeatureStore
from snowpark_session import create_snowpark_session


def setup_entities(session=None) -> tuple:
    """Create Feature Store and register CUSTOMER and TRANSACTION entities."""
    close_session = False
    if session is None:
        session = create_snowpark_session()
        close_session = True

    session.sql(f"USE WAREHOUSE {WAREHOUSE}").collect()

    fs = FeatureStore(
        session=session,
        database=DATABASE,
        name=SCHEMA,
        default_warehouse=WAREHOUSE,
        creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
    )

    customer_entity = Entity(name="CUSTOMER", join_keys=["CUSTOMER_ID"])
    transaction_entity = Entity(name="TRANSACTION", join_keys=["TXN_ID"])

    fs.register_entity(customer_entity)
    fs.register_entity(transaction_entity)

    print("Registered entities:")
    print("  - CUSTOMER (join_key: CUSTOMER_ID)")
    print("  - TRANSACTION (join_key: TXN_ID)")

    if close_session:
        session.close()

    return fs, customer_entity, transaction_entity


if __name__ == "__main__":
    setup_entities()
