"""Tier B: attempt MCP streamable-http initialize + tools/list on remote servers.

Records, per server: reachable? auth-required? tools (name+description) if listable.
"""
import json
import subprocess
import sys

CENSUS_DIR = "/private/tmp/claude-501/-Users-jbarney-Documents-code-prompt-fidelity/99788712-8afe-4cd0-8efa-7473e209fb04/scratchpad/census"

INIT = {
    "jsonrpc": "2.0", "id": 1, "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "pf-census-probe", "version": "0.1"},
    },
}
INITIALIZED = {"jsonrpc": "2.0", "method": "notifications/initialized"}
TOOLS_LIST = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}


def post(url, body, session=None):
    cmd = ["curl", "-s", "-D", "-", "--max-time", "20", "-X", "POST", url,
           "-H", "Content-Type: application/json",
           "-H", "Accept: application/json, text/event-stream",
           "-H", "MCP-Protocol-Version: 2025-06-18",
           "-d", json.dumps(body)]
    if session:
        cmd += ["-H", f"mcp-session-id: {session}"]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=25).stdout.decode("utf-8", "replace")
    except subprocess.TimeoutExpired:
        return None, None, None, "timeout"
    if "\r\n\r\n" not in out and "\n\n" not in out:
        return None, None, None, "no-response"
    # split headers/body (may have multiple header blocks on redirects/continues)
    sep = "\r\n\r\n" if "\r\n\r\n" in out else "\n\n"
    parts = out.split(sep)
    headers, body_txt = "", parts[-1]
    for p in parts[:-1]:
        headers += p + "\n"
    status = None
    sess = session
    for line in headers.splitlines():
        if line.startswith("HTTP/"):
            try:
                status = int(line.split()[1])
            except (IndexError, ValueError):
                pass
        if line.lower().startswith("mcp-session-id:"):
            sess = line.split(":", 1)[1].strip()
    # parse JSON or SSE body
    payload = None
    t = body_txt.strip()
    if t.startswith("{"):
        try:
            payload = json.loads(t)
        except json.JSONDecodeError:
            pass
    else:  # SSE
        for line in t.splitlines():
            if line.startswith("data:"):
                try:
                    payload = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue
    return status, sess, payload, None


def probe(url):
    r = {"url": url, "outcome": None, "tools": []}
    status, sess, payload, err = post(url, INIT)
    if err:
        r["outcome"] = err
        return r
    if status in (401, 403):
        r["outcome"] = "auth-required"
        return r
    if status is None or status >= 400 or not payload or "result" not in payload:
        r["outcome"] = f"init-failed({status})"
        return r
    post(url, INITIALIZED, sess)
    status, sess, payload, err = post(url, TOOLS_LIST, sess)
    if err or not payload or "result" not in payload:
        r["outcome"] = f"tools-list-failed({status or err})"
        return r
    tools = payload["result"].get("tools", [])
    r["outcome"] = "ok"
    r["tools"] = [{"name": t.get("name", ""), "description": (t.get("description") or "")[:300]}
                  for t in tools]
    return r


def main():
    servers = json.load(open(f"{CENSUS_DIR}/servers.json"))
    remotes = [s for s in servers if s["remotes"]]
    results = []
    ok = 0
    target_ok = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    for s in remotes:
        if ok >= target_ok:
            break
        res = probe(s["remotes"][0])
        res["name"] = s["name"]
        res["description"] = s["description"]
        results.append(res)
        if res["outcome"] == "ok" and res["tools"]:
            ok += 1
        print(f"[{ok:3d} ok / {len(results):3d} tried] {s['name']}: {res['outcome']} ({len(res['tools'])} tools)", file=sys.stderr)
    json.dump(results, open(f"{CENSUS_DIR}/tier_b_raw.json", "w"), indent=1)
    outcomes = {}
    for r in results:
        key = r["outcome"] if r["outcome"] in ("ok", "auth-required", "timeout") else "error"
        outcomes[key] = outcomes.get(key, 0) + 1
    print("OUTCOMES:", json.dumps(outcomes), file=sys.stderr)


if __name__ == "__main__":
    main()
