"""Cross-region model promotion strategy.

Promotes a model from a STAGE account to a PROD account in a different
cloud region using Snowflake replication groups or cross-region listings.

NOT YET IMPLEMENTED - placeholder for future work.

Approach options:
  1. Replication group: Replicate database containing model across regions
  2. Cross-region listing: Share model via listing with cross-region fulfillment
  3. Stage-based: Export to cloud storage, import in target region/account

Requirements:
  - Separate OIDC service users per account
  - Cross-region replication enabled on accounts
  - Handling of replication lag (eventual consistency)
  - Region-specific compute pools and gateways in PROD
"""


def promote(version: str, session=None):
    """Promote model version from STAGE account/region to PROD account/region."""
    raise NotImplementedError(
        "Cross-region promotion is not yet implemented. "
        "Set TOPOLOGY=single-account to use cross-database replication within one account."
    )
