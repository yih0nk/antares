"""Which part of the rendered frame throws a LeRobot checkpoint?

Open loop on the lock's *training* episodes (every 5th of them, so the
held-out numbers in ``lerobot_closedloop.json`` are untouched by any of
this): at every 30th frame the checkpoint gets the real joint state and
an image built from a mix of real and rendered parts, and its first 30
actions are scored against the logged commands (body joints, deg RMS).
The composite's arm pool leaves the scored episode out.

Usage (from the repo root)::

    python experiments/lerobot_render_ablation.py --out results/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from antares import server, video
from antares.lock import now_iso
from antares.policy import StepContext, _deg_per_unit, load_lerobot_policy
from antares.simcam import SimCameras, VideoArm

import lerobot_closedloop as X  # noqa: E402 - sibling script


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", default="samarth-shukla6/act-svla-so101-pickplace")
    ap.add_argument("--out", type=Path, default=X.ROOT / "results")
    args = ap.parse_args()

    S = X.Setup()
    pol = load_lerobot_policy(args.checkpoint, task=X.TASK)
    bgm = {n: server.camera_background(S.ds, video.camera_key(n)) for n in X.CAMERAS}
    masks = {n: bgm[n].mask for n in X.CAMERAS}
    tracks = {e.index: S.track(e) + S.offset for e in S.train}
    dpu = _deg_per_unit(S.unit_map)[X.BODY]
    looks = {
        "black arm, unlit brick (default)": {},
        "grey 0.2 arm, brick emission 0.5": {"arm_rgba": (0.2, 0.2, 0.21, 1.0), "object_emission": 0.5},
        "grey 0.3 arm, brick emission 1.0": {"arm_rgba": (0.3, 0.3, 0.32, 1.0), "object_emission": 1.0},
    }
    cams = {k: SimCameras.from_lock(S.lock, X.ROOT / "locks", cameras=X.CAMERAS, **kw) for k, kw in looks.items()}
    base = cams["black arm, unlit brick (default)"]
    errs: dict[str, list[float]] = {}
    eps = S.train[::5]
    for ep in eps:
        va = VideoArm([e for e in S.train if e.index != ep.index], S.unit_map, S.frame, masks, tracks)
        q = S.unit_map.to_rad(ep.state)
        tr = tracks[ep.index]
        for t in range(0, len(ep) - 30, 30):
            real = {n: S.frame(ep.index, n, t) for n in X.CAMERAS}
            rend = base.render(q[t], tr[t])
            comp = base.render(q[t], tr[t], arm=va)
            fg = {}
            for n in X.CAMERAS:  # backdrop + every real foreground pixel (arm and brick) of this very frame
                m = masks[n](real[n]).astype(bool)
                fg[n] = base.backdrops[n].copy()
                fg[n][m] = real[n][m]
            views = {
                "real": real,
                "render": rend,
                "composite": comp,
                "render up, real side": {"up": rend["up"], "side": real["side"]},
                "real up, render side": {"up": real["up"], "side": rend["side"]},
                "sim arm drawn over the real frame": base.render(q[t], None, backgrounds=real),
                "backdrop + real foreground of the same frame": fg,
                "backdrop only": dict(base.backdrops),
            }
            for k, c in cams.items():
                views[f"render: {k}"] = c.render(q[t], tr[t])
                views[f"composite: {k}"] = c.render(q[t], tr[t], arm=va)
            for name, imgs in views.items():
                pol.reset()
                pol.extra_inputs = lambda _c, imgs=imgs: X.as_inputs(imgs)
                chunk = pol.act(ep.state[t : t + 1], StepContext(t, ep.fps))[:30]
                d = (chunk[:, X.BODY] - ep.action[t : t + 30, X.BODY]) * dpu
                errs.setdefault(name, []).append(float(np.sqrt(np.mean(d**2))))
            d = (ep.state[t, X.BODY][None] - ep.action[t : t + 30, X.BODY]) * dpu
            errs.setdefault("hold state (no policy)", []).append(float(np.sqrt(np.mean(d**2))))
    out = {
        "checkpoint": args.checkpoint,
        "dataset": X.DATASET,
        "lock": X.LOCK,
        "episodes": [int(e.index) for e in eps],
        "n_windows": len(errs["real"]),
        "rmse_deg": {k: float(np.mean(v)) for k, v in errs.items()},
        "created_at": now_iso(),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "lerobot_render_ablation.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    for k, v in out["rmse_deg"].items():
        print(f"{v:6.2f}  {k}")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
