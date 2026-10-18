"""Dump reference state/action episodes for every fleet robot into refs.npz.

Needs the Hugging Face Hub, so run it where the Hub is reachable:
``python experiments/dump_refs.py`` writes ``.policy-data/refs.npz`` for
``policy_transfer.py``.
"""
import json, subprocess, sys
from pathlib import Path
import numpy as np
from antares.data import load_dataset

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / ".policy-data" / "refs.npz"
FULL = {"lerobot/svla_so101_pickplace", "lerobot/svla_so100_pickplace"}

p = REPO / "locks/fleet/fleet.json"
text = p.read_text() if p.exists() else subprocess.check_output(
    ["git", "show", "origin/main:locks/fleet/fleet.json"], cwd=REPO, text=True)
robots = json.loads(text)["robots"]

arrays, meta, failed = {}, [], []
for r in robots:
    ds_id, lock = r["dataset"], r["lock"]
    try:
        ds = load_dataset(ds_id)
        eps = sorted(ds.episodes, key=lambda e: e.index)
        eps = eps if ds_id in FULL else eps[:10]
        for e in eps:
            arrays[f"{lock}/{e.index}/state"] = e.state.astype(np.float32)
            arrays[f"{lock}/{e.index}/action"] = e.action.astype(np.float32)
        meta.append({"lock": lock, "dataset": ds_id, "fps": ds.fps,
                     "joint_names": list(ds.joint_names),
                     "episode_indices": [e.index for e in eps]})
        print(f"ok   {ds_id}: {len(eps)} eps, shape[1]={eps[0].state.shape[1]}", flush=True)
    except Exception as exc:
        failed.append((ds_id, f"{type(exc).__name__}: {exc}"))
        print(f"FAIL {ds_id}: {type(exc).__name__}: {exc}", flush=True)

arrays["meta"] = np.array([json.dumps(m) for m in meta])
np.savez_compressed(OUT, **arrays)
print(f"\n{OUT}  {OUT.stat().st_size/1e6:.2f} MB")
print(f"loaded {len(meta)}/{len(robots)} datasets, {sum(len(m['episode_indices']) for m in meta)} episodes")
for d, err in failed:
    print(f"failed: {d}: {err}")
