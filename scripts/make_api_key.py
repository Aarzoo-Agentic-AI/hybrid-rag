"""Create an API key: print it once, store only its SHA-256 (in api.keys_file).

Usage:
    uv run python scripts/make_api_key.py --demo        # employee, manager, hr-comp, admin keys
    uv run python scripts/make_api_key.py alice --user u-1042 --clearance confidential \\
        --group hr-compensation --role people_manager
    uv run python scripts/make_api_key.py ops-admin --admin

The tenant defaults to documents.tenant_id. Keys look like hrk_...; keep them like passwords.
The keys file (api_keys.yaml) is git-ignored. Restart the API after adding keys.
"""

from __future__ import annotations

import argparse
import sys

from hybrid_rag.api.auth import add_key
from hybrid_rag.config import get_settings

DEMO = {
    "demo-employee": {"user_id": "u-demo-employee", "clearance": "internal"},
    "demo-manager": {
        "user_id": "u-demo-manager",
        "clearance": "confidential",
        "roles": ["people_manager"],
    },
    "demo-hr-comp": {
        "user_id": "u-demo-hr",
        "clearance": "confidential",
        "groups": ["hr-compensation"],
    },
    "demo-admin": {"user_id": "u-demo-admin", "clearance": "restricted", "admin": True},
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", nargs="?", help="a label for the key (e.g. the person or app)")
    parser.add_argument("--demo", action="store_true", help="create the four demo keys")
    parser.add_argument("--tenant", help="default: documents.tenant_id")
    parser.add_argument("--user", help="user ID (default: the name)")
    parser.add_argument(
        "--clearance",
        default="internal",
        choices=["public", "internal", "confidential", "restricted"],
    )
    parser.add_argument("--group", action="append", default=[])
    parser.add_argument("--role", action="append", default=[])
    parser.add_argument("--admin", action="store_true", help="tenant admin: ingest, delete, access")
    args = parser.parse_args()

    settings = get_settings()
    path = settings.api.keys_file
    tenant = args.tenant or settings.documents.tenant_id
    if args.demo:
        specs = {name: {"tenant_id": tenant, **spec} for name, spec in DEMO.items()}
    elif args.name:
        spec = {
            "tenant_id": tenant,
            "user_id": args.user or args.name,
            "clearance": "restricted" if args.admin else args.clearance,
            "groups": args.group,
            "roles": args.role,
            "admin": args.admin,
        }
        specs = {args.name: spec}
    else:
        parser.error("give a key name, or --demo")

    print(f"Adding to {path} (only the hashes are stored). Copy the keys now:\n")
    for name, spec in specs.items():
        try:
            key = add_key(path, name, spec)
        except ValueError as exc:
            print(f"  {name}: skipped ({exc})")
            continue
        who = "tenant admin" if spec.get("admin") else f"{spec['clearance']}"
        extras = [*spec.get("groups", []), *spec.get("roles", [])]
        print(f"  {name:<14} {key}   ({who}{', ' + ', '.join(extras) if extras else ''})")
    print("\nSend a key as the X-API-Key header. Restart the API to load new keys.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
