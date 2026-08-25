"""
Snowpark session helper — create a Snowpark Session from local Snowflake CLI config.

Usage as a module (import in your scripts or notebooks):

    from snowpark_session import create_snowpark_session
    session = create_snowpark_session()

Usage as a CLI (test connectivity):

    python snowpark_session.py
    python snowpark_session.py --connection my_conn
    python snowpark_session.py --test

Reads ~/.snowflake/connections.toml (or config.toml fallback) and handles all
authentication methods including private_key_path, externalbrowser, token, etc.

Respects:
    $SNOWFLAKE_HOME           — config directory (default: ~/.snowflake)
    $SNOWFLAKE_CONNECTION_NAME — override connection name
    $SNOWFLAKE_DEFAULT_CONNECTION_NAME — fallback connection name
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

from snowflake.snowpark import Session

# ---------------------------------------------------------------------------
# Keys that Snowpark Session.builder.configs() accepts.
# Unknown keys cause errors, so we filter the TOML config to this set.
# ---------------------------------------------------------------------------
_SNOWPARK_ALLOWED_KEYS = {
    "account",
    "user",
    "password",
    "authenticator",
    "host",
    "port",
    "protocol",
    "role",
    "database",
    "schema",
    "warehouse",
    "token",
    "private_key",
}

# Path to Cortex Code agent settings (contains active connection name)
_AGENT_SETTINGS_PATH = Path("~/.snowflake/cortex/settings.json").expanduser()


# ---------------------------------------------------------------------------
# TOML loading — uses tomllib (Python 3.11+) or tomli (Python 3.10)
# ---------------------------------------------------------------------------
def _load_toml(path: Path) -> dict:
    """Load a TOML file and return its contents as a dict."""
    try:
        import tomllib
    except ModuleNotFoundError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError:
            raise ImportError(
                "TOML parsing requires the 'tomli' package on Python < 3.11. Install it with: pip install tomli"
            )
    with open(path, "rb") as f:
        return tomllib.load(f)


# ---------------------------------------------------------------------------
# Config file discovery
# ---------------------------------------------------------------------------
def _load_all_connections(snowflake_home: Path) -> tuple[dict[str, dict], Optional[str]]:
    """Load all connection configs and default_connection_name from TOML files.

    Priority:
        1. connections.toml (flat structure, each top-level key is a connection)
        2. config.toml (connections under [connections] section)

    Returns:
        (connections_dict, default_connection_name_or_None)
    """
    connections_path = snowflake_home / "connections.toml"
    config_path = snowflake_home / "config.toml"

    if connections_path.exists():
        data = _load_toml(connections_path)
        default_name = data.get("default_connection_name")
        connections = {k: v for k, v in data.items() if k != "default_connection_name" and isinstance(v, dict)}
        # Also check config.toml for default_connection_name if not in connections.toml
        if default_name is None and config_path.exists():
            cfg = _load_toml(config_path)
            default_name = cfg.get("default_connection_name")
        return connections, default_name

    if config_path.exists():
        data = _load_toml(config_path)
        default_name = data.get("default_connection_name")
        connections = data.get("connections", {})
        return connections, default_name

    raise FileNotFoundError(
        f"No connections.toml or config.toml found in {snowflake_home}. "
        f"Configure a connection with: snow connection add"
    )


# ---------------------------------------------------------------------------
# Connection name resolution
# ---------------------------------------------------------------------------
def _read_agent_connection_name() -> Optional[str]:
    """Read the active connection name from Cortex Code agent settings."""
    if not _AGENT_SETTINGS_PATH.exists():
        return None
    try:
        data = json.loads(_AGENT_SETTINGS_PATH.read_text())
        return data.get("cortexAgentConnectionName")
    except (json.JSONDecodeError, OSError):
        return None


def _resolve_connection_name(
    explicit: Optional[str],
    default_from_toml: Optional[str],
    available: list[str],
) -> str:
    """Resolve which connection name to use.

    Priority: explicit arg > $SNOWFLAKE_CONNECTION_NAME >
              $SNOWFLAKE_DEFAULT_CONNECTION_NAME > TOML default >
              agent settings > first available connection.
    """
    name = (
        explicit
        or os.getenv("SNOWFLAKE_CONNECTION_NAME")
        or os.getenv("SNOWFLAKE_DEFAULT_CONNECTION_NAME")
        or default_from_toml
        or _read_agent_connection_name()
    )
    if name and name in available:
        return name
    if name and name not in available:
        raise KeyError(f"Connection '{name}' not found. Available: {available}")
    # Fall back to first available
    if available:
        return available[0]
    raise KeyError("No connections found in Snowflake config files.")


# ---------------------------------------------------------------------------
# Auth handling
# ---------------------------------------------------------------------------
def _resolve_private_key(config: dict) -> dict:
    """Load private key from file path if specified, handling PEM and DER formats."""
    pk_path = (
        config.pop("private_key_path", None)
        or config.pop("private_key_file", None)
        or config.pop("privatekeypath", None)
    )
    if not pk_path:
        return config

    from cryptography.hazmat.primitives import serialization

    key_path = Path(pk_path).expanduser()
    if not key_path.exists():
        raise FileNotFoundError(
            f"Private key file not found: {key_path}. Check private_key_path in your Snowflake connection config."
        )

    passphrase = config.pop("private_key_passphrase", None)
    password = passphrase.encode() if passphrase else None
    key_data = key_path.read_bytes()

    # Detect PEM vs DER format
    if b"-----BEGIN" in key_data:
        private_key = serialization.load_pem_private_key(key_data, password=password)
    else:
        private_key = serialization.load_der_private_key(key_data, password=password)

    config["private_key"] = private_key
    return config


def _resolve_token_file(config: dict) -> dict:
    """Read token from token_file_path if specified (used in SPCS / container environments)."""
    token_file = config.pop("token_file_path", None)
    if token_file and not config.get("token"):
        token_path = Path(token_file)
        if token_path.exists():
            config["token"] = token_path.read_text().strip()
    return config


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------
def create_snowpark_session(connection_name: Optional[str] = None) -> Session:
    """Create a Snowpark session from local Snowflake CLI config files.

    Handles all authentication methods (password, externalbrowser, private_key,
    token, etc.) and filters config to only keys that Snowpark accepts.

    In CI environments (GitHub Actions with OIDC), detects SNOWFLAKE_TOKEN and
    SNOWFLAKE_AUTHENTICATOR env vars and uses those directly.

    Args:
        connection_name: Explicit connection name. If None, resolved from env
            vars, TOML defaults, or agent settings.

    Returns:
        A connected Snowpark Session.
    """
    # CI/OIDC detection: if SNOWFLAKE_TOKEN and SNOWFLAKE_ACCOUNT are set,
    # we're running in GitHub Actions with snowflake-actions@v3 OIDC auth.
    if os.getenv("SNOWFLAKE_TOKEN") and os.getenv("SNOWFLAKE_ACCOUNT"):
        config = {
            "account": os.environ["SNOWFLAKE_ACCOUNT"],
            "token": os.environ["SNOWFLAKE_TOKEN"],
            "authenticator": os.getenv("SNOWFLAKE_AUTHENTICATOR", "oauth"),
        }
        if os.getenv("SNOWFLAKE_WORKLOAD_IDENTITY_PROVIDER"):
            config["workload_identity_provider"] = os.environ["SNOWFLAKE_WORKLOAD_IDENTITY_PROVIDER"]
        if os.getenv("SNOWFLAKE_AUDIENCE"):
            config["audience"] = os.environ["SNOWFLAKE_AUDIENCE"]
        if os.getenv("SNOWFLAKE_USER"):
            config["user"] = os.environ["SNOWFLAKE_USER"]
        if os.getenv("SNOWFLAKE_DATABASE"):
            config["database"] = os.environ["SNOWFLAKE_DATABASE"]
        if os.getenv("SNOWFLAKE_SCHEMA"):
            config["schema"] = os.environ["SNOWFLAKE_SCHEMA"]
        if os.getenv("SNOWFLAKE_WAREHOUSE"):
            config["warehouse"] = os.environ["SNOWFLAKE_WAREHOUSE"]
        if os.getenv("SNOWFLAKE_ROLE"):
            config["role"] = os.environ["SNOWFLAKE_ROLE"]
        return Session.builder.configs(config).create()

    snowflake_home = Path(os.environ.get("SNOWFLAKE_HOME", "~/.snowflake")).expanduser()

    all_connections, default_name = _load_all_connections(snowflake_home)

    conn_name = _resolve_connection_name(
        explicit=connection_name,
        default_from_toml=default_name,
        available=list(all_connections.keys()),
    )

    raw_config = dict(all_connections[conn_name])

    # Handle auth-specific keys before filtering
    raw_config = _resolve_private_key(raw_config)
    raw_config = _resolve_token_file(raw_config)

    # Filter to only keys Snowpark accepts — prevents errors from unknown keys
    config = {k: v for k, v in raw_config.items() if k in _SNOWPARK_ALLOWED_KEYS}

    return Session.builder.configs(config).create()


# ---------------------------------------------------------------------------
# CLI entry point — for testing connectivity
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Test Snowflake Snowpark connectivity using local config.")
    parser.add_argument(
        "--connection",
        "-c",
        help="Connection name from connections.toml / config.toml",
    )
    parser.add_argument(
        "--test",
        "-t",
        action="store_true",
        help="Run a test query (SELECT CURRENT_USER(), CURRENT_ROLE())",
    )
    args = parser.parse_args()

    try:
        print("Creating Snowpark session...")
        session = create_snowpark_session(connection_name=args.connection)
        print("✅ Connected successfully!")
        print(f"   Account:   {session.get_current_account()}")
        print(f"   User:      {session.get_current_user()}")
        print(f"   Role:      {session.get_current_role()}")
        print(f"   Database:  {session.get_current_database()}")
        print(f"   Schema:    {session.get_current_schema()}")
        print(f"   Warehouse: {session.get_current_warehouse()}")

        if args.test:
            print("\nRunning test query: SELECT 1 AS test_col")
            result = session.sql("SELECT 1 AS test_col").collect()
            print(f"   Result: {result}")

        session.close()
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
