"""Multi-account model promotion strategy (same region).

Promotes a model from a STAGE account to a PROD account using
cross-account model sharing or Snowflake listings.

NOT YET IMPLEMENTED - placeholder for future work.

Approach options:
  1. Private listing: Share model via a private listing from STAGE account
  2. Direct share: CREATE MODEL ... FROM MODEL using cross-account reference
  3. Export/import: Export model artifact to shared stage, import in PROD account

Requirements:
  - Separate OIDC service users per account
  - Network connectivity between accounts (same region)
  - Appropriate grants for cross-account object access
"""


def promote(version: str, session=None):
    """Promote model version from STAGE account to PROD account."""
    raise NotImplementedError(
        "Multi-account promotion is not yet implemented. "
        "Set TOPOLOGY=single-account to use cross-database replication within one account."
    )
