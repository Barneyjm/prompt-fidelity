"""Paginate the MCP registry, dedupe to latest version per server name."""
import json
import subprocess
import sys
import time
import urllib.parse

BASE = "https://registry.modelcontextprotocol.io/v0.1/servers"
TARGET = 1000
OUT = sys.argv[1] if len(sys.argv) > 1 else "servers.json"

servers = {}  # name -> entry (latest version wins)
cursor = None
pages = 0
total_entries = 0

while len(servers) < TARGET:
    url = BASE + "?limit=100"
    if cursor:
        url += "&cursor=" + urllib.parse.quote(cursor)
    raw = subprocess.run(["curl", "-s", "--max-time", "30", url],
                         capture_output=True, check=True).stdout
    data = json.loads(raw)
    pages += 1
    batch = data.get("servers", [])
    total_entries += len(batch)
    for item in batch:
        s = item.get("server", {})
        meta = item.get("_meta", {}).get("io.modelcontextprotocol.registry/official", {})
        name = s.get("name")
        if not name:
            continue
        # later pages iterate versions in order; isLatest marks the definitive one,
        # but always keep the most recent seen so non-latest-only names still appear
        if name not in servers or meta.get("isLatest"):
            servers[name] = {
                "name": name,
                "title": s.get("title", ""),
                "description": (s.get("description") or "").strip(),
                "version": s.get("version", ""),
                "status": meta.get("status", ""),
                "remotes": [r.get("url") for r in s.get("remotes", []) if r.get("url")],
                "has_remote": bool(s.get("remotes")),
                "publishedAt": meta.get("publishedAt", ""),
            }
    cursor = data.get("metadata", {}).get("nextCursor")
    if pages % 10 == 0:
        print(f"pages={pages} entries={total_entries} unique={len(servers)}", file=sys.stderr)
    if not cursor:
        break
    time.sleep(0.15)

active = [v for v in servers.values() if v["status"] in ("active", "")]
print(f"DONE pages={pages} raw_entries={total_entries} unique_names={len(servers)} active={len(active)}", file=sys.stderr)
with open(OUT, "w") as f:
    json.dump(active, f, indent=1)
