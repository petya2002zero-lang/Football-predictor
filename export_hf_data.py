"""
export_hf_data.py — run by GitHub Actions before HF sync.
Exports kv_store from database.sqlite → hf_data.json.
HF Spaces gets the JSON instead of the SQLite file (which exceeds the 10 MB limit).
"""
import json
import os
import sqlite3

WORKSPACE = os.environ.get("GITHUB_WORKSPACE", ".")
db_path   = os.path.join(WORKSPACE, "database.sqlite")
out_path  = os.path.join(WORKSPACE, "hf_data.json")

try:
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT key, value FROM kv_store").fetchall()
    conn.close()
    # bet_log is private user data — never pushed to HF
    data = {k: json.loads(v) for k, v in rows if k not in ("bet_log",)}
    with open(out_path, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    size_kb = os.path.getsize(out_path) // 1024
    print(f"hf_data.json: {len(data)} keys, {size_kb} KB")
except Exception as e:
    print(f"WARNING: export failed ({e}) — writing empty hf_data.json")
    with open(out_path, "w") as f:
        f.write("{}")
