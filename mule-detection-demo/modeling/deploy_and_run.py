#!/usr/bin/env python3
"""
Deploy the modeling/ notebooks to Databricks and run them sequentially.

Usage (from this folder):
    python deploy_and_run.py                       # deploy + run all
    python deploy_and_run.py --only 01,02          # deploy + run a subset
    python deploy_and_run.py --skip-deploy         # just run, don't re-upload
    python deploy_and_run.py --skip-run            # upload only
    python deploy_and_run.py --workspace-folder /Workspace/...

Reads the same `config.py` the notebooks read, so a re-user only changes that file.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
CONFIG_PATH = HERE / "config.py"

# Order matters: 00 must run first; 10 last.
NOTEBOOKS = [
    "00_data_generation.py",
    "01_rules_engine.py",
    "02_isolation_forest.py",
    "03_autoencoder.py",
    "04_xgboost_pu_learning.py",
    "05_graphframes_features.py",
    "06_graphsage_gnn.py",
    "07_lstm_sequence.py",
    "08_muletrack_markov.py",
    "09_tgn_temporal_graph.py",
    "10_tier_comparison.py",
]

# Shared notebooks that get uploaded but never submitted as jobs.
SHARED = ["config.py", "_shared.py"]


# ----------------------------------------------------------------------------
# Tiny config parser — pulls CLUSTER_ID / WORKSPACE_FOLDER out of config.py
# without needing to import it (config.py uses Databricks magic commands).
# ----------------------------------------------------------------------------
def _parse_config_var(name: str) -> str:
    text = CONFIG_PATH.read_text()
    m = re.search(rf'^{name}\s*=\s*[\'"]([^\'"]+)[\'"]', text, flags=re.M)
    if not m:
        raise RuntimeError(f"Could not find {name} in {CONFIG_PATH}")
    return m.group(1)


def databricks(*args: str, json_in: dict | None = None) -> subprocess.CompletedProcess:
    """Call the databricks CLI; return CompletedProcess."""
    cmd = ["databricks", *args]
    if json_in is not None:
        cmd.extend(["--json", json.dumps(json_in)])
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"$ {' '.join(cmd)}\nSTDOUT: {proc.stdout}\nSTDERR: {proc.stderr}",
              file=sys.stderr)
    return proc


def upload_one(local: Path, workspace_path: str) -> None:
    proc = databricks(
        "workspace", "import", workspace_path,
        "--file",     str(local),
        "--format",   "SOURCE",
        "--language", "PYTHON",
        "--overwrite",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to upload {local.name}")
    print(f"  ↑ {local.name}  →  {workspace_path}")


def submit_run(cluster_id: str, notebook_path: str, run_name: str) -> str:
    """Submit a one-off job run asynchronously. Returns the run_id immediately."""
    payload = {
        "run_name": run_name,
        "tasks": [{
            "task_key":            "main",
            "existing_cluster_id": cluster_id,
            "notebook_task":       {"notebook_path": notebook_path},
        }],
    }
    # --no-wait makes the CLI return as soon as the job is queued (does not poll).
    # Our own poll_run() handles waiting for completion.
    proc = databricks("jobs", "submit", "--no-wait", json_in=payload)
    if proc.returncode != 0:
        raise RuntimeError(f"Submit failed for {notebook_path}")
    run_id = json.loads(proc.stdout)["run_id"]
    return str(run_id)


def poll_run(run_id: str, poll_s: int = 15) -> dict:
    """Block until the run terminates. Returns the final run info dict."""
    while True:
        proc = databricks("jobs", "get-run", run_id)
        if proc.returncode != 0:
            raise RuntimeError(f"get-run failed for {run_id}")
        info = json.loads(proc.stdout)
        state = info.get("state", {}).get("life_cycle_state", "PENDING")
        if state in ("TERMINATED", "INTERNAL_ERROR", "SKIPPED"):
            return info
        print(f"    … run {run_id} state={state}", flush=True)
        time.sleep(poll_s)


# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", help="comma-separated notebook prefixes (e.g. 01,02)")
    p.add_argument("--skip-deploy", action="store_true")
    p.add_argument("--skip-run",    action="store_true")
    p.add_argument("--workspace-folder", default=None,
                   help="Override WORKSPACE_FOLDER from config.py")
    args = p.parse_args()

    workspace_folder = args.workspace_folder or _parse_config_var("WORKSPACE_FOLDER")
    cluster_id       = _parse_config_var("CLUSTER_ID")
    print(f"Workspace folder : {workspace_folder}")
    print(f"Cluster id       : {cluster_id}")

    only = set((args.only or "").split(",")) if args.only else None
    selected = [
        nb for nb in NOTEBOOKS
        if not only or nb.split("_")[0] in only
    ]
    print(f"Selected notebooks ({len(selected)}): {[s.split('_')[0] for s in selected]}")

    # --- Phase A: upload --------------------------------------------------
    if not args.skip_deploy:
        print("\n=== Upload ===")
        databricks("workspace", "mkdirs", workspace_folder)
        for fname in SHARED + selected:
            local = HERE / fname
            if not local.exists():
                print(f"  ✗ missing {fname}, skipping"); continue
            ws_name = fname.removesuffix(".py")
            upload_one(local, f"{workspace_folder}/{ws_name}")

    # --- Phase B: run sequentially ----------------------------------------
    if args.skip_run:
        return

    print("\n=== Run ===")
    results = []
    for fname in selected:
        ws_name = fname.removesuffix(".py")
        ws_path = f"{workspace_folder}/{ws_name}"
        print(f"\n▶ submitting {fname}")
        t0 = time.time()
        try:
            run_id = submit_run(cluster_id, ws_path, run_name=f"mule-demo {ws_name}")
            info  = poll_run(run_id)
        except RuntimeError as e:
            # The CLI exits non-zero when the run finishes in any non-success state.
            # Treat as a failed run and keep going.
            dur = time.time() - t0
            print(f"  ✗ {fname}  result=ERROR  ({dur:.0f}s)  {e}")
            results.append((ws_name, "ERROR", dur, "", str(e)))
            if ws_name.startswith("00") or ws_name == "_shared":
                print("    ⚠  foundational notebook failed; aborting the rest")
                break
            continue
        dur   = time.time() - t0
        state = info.get("state", {}); result_state = state.get("result_state", "UNKNOWN")
        msg   = state.get("state_message") or state.get("result_message", "")
        url   = info.get("run_page_url", "")
        ok    = result_state == "SUCCESS"
        results.append((ws_name, result_state, dur, url, msg))
        print(f"  {'✓' if ok else '✗'} {fname}  result={result_state}  "
              f"({dur:.0f}s)  {url}")
        if not ok:
            print(f"    message: {msg}")
            if ws_name.startswith("00") or ws_name == "_shared":
                print("    ⚠  foundational notebook failed; aborting the rest")
                break

    # --- Summary ---------------------------------------------------------
    print("\n=== Summary ===")
    print(f"{'notebook':30s} {'result':12s} {'dur':>6s}  url")
    for ws_name, res, dur, url, _ in results:
        print(f"{ws_name:30s} {res:12s} {dur:>5.0f}s  {url}")


if __name__ == "__main__":
    main()
