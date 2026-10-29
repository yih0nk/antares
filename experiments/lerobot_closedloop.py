"""Real LeRobot checkpoints, closed loop, in the stock and the calibrated sim.

For each Hub checkpoint trained on ``lerobot/svla_so101_pickplace`` (ACT,
SmolVLA) and each of the lock's held-out episodes:

1. **Does the checkpoint work here, and what does the render cost it?**
   (open loop, real joint states) At every ``--stride``-th frame the
   checkpoint is fed the real joint state with the real ``up``/``side``
   video frames and predicts a chunk; its first ``--horizon`` actions are
   scored against the logged commands (deg RMS, body joints). The same
   states are then fed with the frames *rendered* by :mod:`antares.simcam`
   (the fitted cameras, the scene backdrops, the object where the logs put
   it), once with MuJoCo's meshes for the arm (``render``) and once with
   the real arm's pixels from the nearest recorded pose of the lock's
   *training* episodes (``composite``). "Hold state" (repeat the current
   state for the whole chunk) is the floor: a policy that cannot beat it
   is not reading anything useful from the image.
2. **Closed loop.** The checkpoint drives the default sim and the lock,
   seeing only rendered frames and the sim's joint state; the object is a
   :class:`antares.task.KinematicObject` at the operator's grasp point,
   and success is lift + release within ``PLACE_CM`` of where the operator
   released it. ``underlay`` puts the real video frame of the same instant
   under the sim arm instead of the backdrop: a diagnostic (it leaks the
   real episode), not an evaluation.
3. **Control.** The logged commands replayed closed-loop through the same
   object and check: what the check says about the operator.

Usage (from the repo root; needs ``pip install lerobot`` and the Hub)::

    python experiments/lerobot_closedloop.py --out results/ \\
        --checkpoint samarth-shukla6/act-svla-so101-pickplace
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from antares import server, video
from antares.closedloop import closed_loop_episodes, run_closed_loop
from antares.data import Episode
from antares.lock import now_iso
from antares.policy import LoggedPolicy, StepContext, _deg_per_unit, load_lerobot_policy
from antares.scene import load_scene_fit
from antares.shadow import SimVariant
from antares.simcam import SimCameras, VideoArm
from antares.task import LIFT_M, PICK_RADIUS_M, PLACE_CM, KinematicObject, find_grasps, place_target

ROOT = Path(__file__).resolve().parents[1]
DATASET = "lerobot/svla_so101_pickplace"
LOCK = "svla_so101_pickplace"
TASK = "pink lego brick into the transparent box"  # the dataset's meta/tasks.parquet
CAMERAS = ("up", "side")
BODY = [0, 1, 2, 3, 4]


class Setup:
    """Everything the runs share: dataset, split, sims, objects, cameras."""

    def __init__(self, episodes: str = ""):
        self.ds = server.get_dataset(DATASET)
        self.lock = server.load_lock_dict(LOCK)
        self.train, self.held = server.policy_split(self.ds, self.lock)
        if episodes:
            keep = {int(x) for x in episodes.split(",")}
            self.held = [e for e in self.held if e.index in keep]
        self.variants, _ = server.resolve_policy_variants(self.lock, ["default", "lock"])
        self.ctx = server.grasp_context_for(self.ds, self.lock)
        self.unit_map = self.ctx.unit_map
        c = self.ctx
        gaps = [g.jaw_gap for ep in self.train for g in find_grasps(ep, c.thresholds, c.tip, c.unit_map, c.empty) if g.kind == "pick"]
        self.held_jaw = float(c.empty.jaw + np.median(gaps)) if gaps else None
        self.info: dict[int, dict[str, Any]] = {}
        self.cams = SimCameras.from_lock(self.lock, ROOT / "locks", cameras=CAMERAS)
        # the loop tracks the fingertip at the grasp; the rendered box rests on the table
        table_z = load_scene_fit(LOCK, ROOT / "locks").table.z
        rest = [o.rest_z for o in (self.objects(e) for e in self.train) if o is not None]
        self.offset = np.array([0.0, 0.0, table_z + 0.5 * self.cams.object_size[2] - float(np.median(rest))])
        self.lengths = {int(e.index): len(e) for e in self.ds.episodes}
        self._frames: dict[tuple[int, str, int], np.ndarray] = {}
        bgm = {n: server.camera_background(self.ds, video.camera_key(n)) for n in CAMERAS}
        tracks = {int(e.index): self.track(e) + self.offset for e in self.train}
        self.video_arm = VideoArm(self.train, self.unit_map, self.frame, {n: bgm[n].mask for n in CAMERAS}, tracks)

    def objects(self, ep: Episode) -> KinematicObject | None:
        """The episode's object at the operator's first pick, placed where the operator let go."""
        c = self.ctx
        grasps = find_grasps(ep, c.thresholds, c.tip, c.unit_map, c.empty)
        picks = [g for g in grasps if g.kind == "pick"]
        if not picks:
            return None
        place = place_target(grasps)
        self.info[int(ep.index)] = {
            "start_xyz": list(picks[0].xyz),
            "place_xyz": None if place is None else place.tolist(),
            "real_pick_frame": picks[0].frame,
            "real_release_frame": picks[0].release_frame,
        }
        return KinematicObject(picks[0].xyz, c.thresholds, c.tip, place_xyz=place, held_jaw=self.held_jaw)

    def track(self, ep: Episode) -> np.ndarray:
        """The object driven by the real episode: (T,3), where the logs say it was."""
        obj = self.objects(ep)
        q = self.unit_map.to_rad(ep.state)
        if obj is None:
            return np.full((len(ep), 3), np.nan)
        return np.stack([obj.step(t, q[t], float(ep.action[t, 5])) for t in range(len(ep))])

    def frame(self, ep: int, cam: str, t: int) -> np.ndarray:
        t = int(np.clip(t, 0, self.lengths[ep] - 1))
        key = (ep, cam, t)
        if key not in self._frames:
            self._frames[key] = video.frames(self.ds.repo_id, video.camera_key(cam), ep, [t], self.ds)[0]
        return self._frames[key]


def as_inputs(imgs: dict[str, np.ndarray]) -> dict[str, Any]:
    import torch

    return {
        f"observation.images.{n}": torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).float().div(255.0)
        for n, im in imgs.items()
    }


def open_loop(policy: Any, S: Setup, stride: int, horizon: int) -> dict[str, Any]:
    """Real states; chunk vs logged commands with real, rendered and composited frames."""
    dpu = _deg_per_unit(S.unit_map)[BODY]
    kinds = ("real", "render", "composite", "render_no_object", "hold_state")
    errs: dict[str, list[float]] = {k: [] for k in kinds}
    per_ep = []
    S.video_arm.distances.clear()
    for ep in S.held:
        tr = S.track(ep) + S.offset
        q = S.unit_map.to_rad(ep.state)
        e_ep: dict[str, list[float]] = {k: [] for k in kinds}
        for t in range(0, len(ep) - horizon, stride):
            target = ep.action[t : t + horizon, BODY]
            o = None if np.isnan(tr[t]).any() else tr[t]
            views = {
                "real": {n: S.frame(ep.index, n, t) for n in CAMERAS},
                "render": S.cams.render(q[t], o),
                "composite": S.cams.render(q[t], o, arm=S.video_arm),
                "render_no_object": S.cams.render(q[t], None),
            }
            for name, imgs in views.items():
                policy.reset()
                policy.extra_inputs = lambda c, imgs=imgs: as_inputs(imgs)
                chunk = policy.act(ep.state[t : t + 1], StepContext(t, ep.fps))[:horizon]
                d = (chunk[:, BODY] - target[: len(chunk)]) * dpu
                e_ep[name].append(float(np.sqrt(np.mean(d**2))))
            d = (ep.state[t, BODY][None] - target) * dpu
            e_ep["hold_state"].append(float(np.sqrt(np.mean(d**2))))
        for k in kinds:
            errs[k] += e_ep[k]
        per_ep.append({"episode": int(ep.index), **{k: float(np.mean(v)) for k, v in e_ep.items()}})
    nn = np.asarray(S.video_arm.distances)
    return {
        "stride": stride,
        "horizon": horizon,
        "n_windows": len(errs["real"]),
        "rmse_deg": {k: float(np.mean(v)) for k, v in errs.items()},
        "composite_pose_error_deg": {"median": float(np.median(nn)), "max": float(nn.max())} if nn.size else None,
        "per_episode": per_ep,
    }


def logged_control(S: Setup) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for v in S.variants:
        arm = v.arm()
        rows = []
        for ep in S.held:
            obj = S.objects(ep)
            if obj is None:
                continue
            run_closed_loop(LoggedPolicy(ep, k=1, h=1), arm, arm.real_q(ep)[0], len(ep), ep.fps, obj=obj, episode=int(ep.index))
            rows.append({"episode": int(ep.index), **obj.result().to_dict()})
        out[v.name] = {
            "n": len(rows),
            "n_picked": sum(r["picked"] for r in rows),
            "n_lifted": sum(r["lifted"] for r in rows),
            "n_placed": sum(r["success"] for r in rows),
            "runs": rows,
        }
    return out


def save_views(path: Path, S: Setup, ep: Episode, sim_q: np.ndarray, trace: list[np.ndarray], ts: list[int],
               arm: VideoArm | None) -> None:
    """Real frame | what the policy saw in closed loop, one row per (frame, camera)."""
    import cv2

    rows = []
    for t in ts:
        r = S.cams.render(sim_q[t], trace[t], arm=arm)
        for n in CAMERAS:
            rows.append(np.hstack([S.frame(ep.index, n, t), r[n]]))
    img = cv2.resize(np.vstack(rows), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 80])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", action="append", required=True, help="Hub id or directory (repeat)")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    ap.add_argument("--episodes", type=str, default="", help="comma-separated subset of the held-out split")
    ap.add_argument("--stride", type=int, default=30, help="open-loop query every N frames")
    ap.add_argument("--horizon", type=int, default=30, help="open-loop actions scored per query")
    ap.add_argument("--modes", type=str, default="render,composite,underlay")
    ap.add_argument("--no-open-loop", action="store_true")
    args = ap.parse_args()

    S = Setup(args.episodes)
    out_path = args.out / "lerobot_closedloop.json"
    results = json.loads(out_path.read_text()) if out_path.is_file() else {}
    for e in S.held:
        S.objects(e)
    results.update(
        dataset=DATASET, lock=LOCK, task=TASK, held_out_episodes=[e.index for e in S.held],
        composite_pool_episodes=[e.index for e in S.train],
        object={
            "pick_radius_m": PICK_RADIUS_M, "lift_m": LIFT_M, "place_cm": PLACE_CM, "held_jaw_units": S.held_jaw,
            "size_m": list(S.cams.object_size), "render_z_offset_m": float(S.offset[2]),
            "episodes": {str(e.index): S.info[e.index] for e in S.held if e.index in S.info},
        },
    )
    t0 = time.time()
    results["logged_control"] = logged_control(S)
    print("control:", {k: {kk: v[kk] for kk in ("n", "n_picked", "n_lifted", "n_placed")} for k, v in results["logged_control"].items()},
          f"[{time.time() - t0:.0f}s]", flush=True)

    ckpts = results.setdefault("checkpoints", {})
    for cid in args.checkpoint:
        t0 = time.time()
        pol = load_lerobot_policy(cid, task=TASK)
        cfg = pol.policy.config
        entry: dict[str, Any] = dict(ckpts.get(cid) or {})
        entry.update(type=cfg.type, n_action_steps=pol.h, chunk_size=getattr(cfg, "chunk_size", None), image_keys=pol.image_keys)
        print(f"{cid}: {cfg.type}, h={pol.h}, images {pol.image_keys}", flush=True)
        if not args.no_open_loop:
            entry["open_loop"] = open_loop(pol, S, args.stride, args.horizon)
            print(f"  open loop: {entry['open_loop']['rmse_deg']} [{time.time() - t0:.0f}s]", flush=True)
        cl = entry.setdefault("closed_loop", {})
        for mode in [m for m in args.modes.split(",") if m]:
            t1 = time.time()
            bg = None
            if mode == "underlay":
                def bg(c, n):
                    return S.frame(c.episode, n, c.frame)
            arm = S.video_arm if mode == "composite" else None
            S.video_arm.distances.clear()
            pol.extra_inputs = S.cams.policy_inputs(CAMERAS, background=bg, object_offset=S.offset, arm=arm)
            s = closed_loop_episodes(pol, S.held, S.variants, S.ctx, objects=S.objects)
            d = s.to_dict(include_arrays=False)
            cl[mode] = {"summary": d["summary"], "runs": d["runs"], "run_names": d["run_names"]}
            if arm is not None and arm.distances:
                cl[mode]["composite_pose_error_deg"] = {"median": float(np.median(arm.distances)), "max": float(np.max(arm.distances))}
            print(f"  closed loop {mode} [{time.time() - t1:.0f}s]:", flush=True)
            for n, v in d["summary"]["variants"].items():
                print(f"    {n:12s} placed {v.get('n_placed')}/{v.get('n_objects')} lifted {v.get('n_lifted')} picked {v.get('n_picked')}"
                      f"  approach {v['approach_rms_deg']:.1f} deg  closest {v['min_tip_distance_cm']:.1f} cm"
                      f"  closed {v['n_closed']}/{v['n_judged']}", flush=True)
            if mode in ("render", "composite"):
                ep = S.held[0]
                lv = next(v for v in S.variants if v.name == "lock")
                a, obj = lv.arm(), S.objects(ep)
                sim_q, _ = run_closed_loop(pol, a, a.real_q(ep)[0], len(ep), ep.fps, obj=obj, episode=int(ep.index))
                trace = [obj.start + S.offset] + [p + S.offset for p in obj.trace]
                save_views(args.out / "lerobot_closedloop" / f"{cid.replace('/', '__')}_{mode}_ep{ep.index}.jpg", S, ep,
                           sim_q, trace, [0, len(ep) // 3, 2 * len(ep) // 3], arm)
        entry["elapsed_s"] = time.time() - t0
        ckpts[cid] = entry
        results["created_at"] = now_iso()
        args.out.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
        print(f"  saved {out_path} [{entry['elapsed_s']:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
