"""Verify the CI OIDC token-minting path without needing a live GitHub runner.

Checks three things:
  1. Outside Actions (no ACTIONS_ID_TOKEN_REQUEST_* vars) minting returns None so
     the existing SNOWFLAKE_TOKEN is used unchanged.
  2. Inside a simulated Actions environment, a fresh token is requested from the
     local stub and that value -- not the stale env var -- ends up in the config.
  3. If the token endpoint fails, it falls back to SNOWFLAKE_TOKEN instead of
     raising, so a transient mint failure does not break the job outright.
"""

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "source"))
import snowpark_session as ss

FRESH = "fresh-token-from-endpoint"


class Handler(BaseHTTPRequestHandler):
    fail = False

    def do_GET(self):
        if Handler.fail:
            self.send_response(500)
            self.end_headers()
            return
        assert self.headers.get("Authorization") == "Bearer request-token", "missing bearer auth"
        assert "audience=snowflakecomputing.com" in self.path, f"audience not passed: {self.path}"
        body = json.dumps({"value": FRESH}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def main():
    failures = []

    # 1. Not in Actions -> no minting, stale token preserved
    for var in ("ACTIONS_ID_TOKEN_REQUEST_URL", "ACTIONS_ID_TOKEN_REQUEST_TOKEN"):
        os.environ.pop(var, None)
    os.environ.update(SNOWFLAKE_ACCOUNT="ACCT", SNOWFLAKE_TOKEN="stale", SNOWFLAKE_ROLE="MLOPS_DEPLOY_ROLE")
    if ss._mint_github_oidc_token() is not None:
        failures.append("expected None outside Actions")
    cfg = ss._ci_oidc_config()
    if cfg["token"] != "stale":
        failures.append(f"expected stale token outside Actions, got {cfg['token']}")
    if cfg.get("role") != "MLOPS_DEPLOY_ROLE":
        failures.append("role not propagated")
    print(f"  1. outside Actions      -> token={cfg['token']!r} (stale preserved)")

    # 2. Simulated Actions -> fresh token replaces the stale one
    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}/token?api-version=2.0"
    os.environ["ACTIONS_ID_TOKEN_REQUEST_URL"] = base
    os.environ["ACTIONS_ID_TOKEN_REQUEST_TOKEN"] = "request-token"

    minted = ss._mint_github_oidc_token()
    if minted != FRESH:
        failures.append(f"expected {FRESH!r}, got {minted!r}")
    cfg = ss._ci_oidc_config()
    if cfg["token"] != FRESH:
        failures.append(f"config should use the fresh token, got {cfg['token']!r}")
    print(f"  2. inside Actions       -> token={cfg['token']!r} (stale overridden)")

    # 3. Endpoint failure -> graceful fallback, no exception
    Handler.fail = True
    cfg = ss._ci_oidc_config()
    if cfg["token"] != "stale":
        failures.append(f"expected fallback to stale on error, got {cfg['token']!r}")
    print(f"  3. endpoint failing     -> token={cfg['token']!r} (graceful fallback)")

    server.shutdown()

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\nAll OIDC minting assertions passed.")


if __name__ == "__main__":
    main()
