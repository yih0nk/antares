"""Which MJCF should simulate SO-100 datasets? The README's Supported robots table.

Scores the committed ``svla_so100_pickplace`` lock's 10 held-out episodes,
read in degrees, on three models: the SO-101 file (``so101_deg``), the
SO-100 Menagerie file verbatim (Euler at 2 ms), and the SO-100 file with
the SO-101 file's ``<option>`` block (implicitfast at 5 ms). For each it
reports the default parameters, the committed lock's parameters and a fresh
150-evaluation calibration (seed 0) on the lock's own split. Needs the Hub::

    .venv/bin/python scripts/so100_model_choice.py --out results/so100_model_choice.json
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path

from antares.calibrate import calibrate, default_simulate, score
from antares.data import load_dataset
from antares.lock import RobotLock
from antares.robots import REGISTRY, register, variant
from antares.sim import SimParams

ROOT = Path(__file__).resolve().parent.parent
DEGREES = dict(unit_map_factory=None, units=("degrees",) * 5 + ("range_0_100",))


def so100_with_so101_options(tmp: Path) -> Path:
    """The SO-100 MJCF with the SO-101 file's ``<option>``, next to a link to its meshes."""
    so100, so101 = REGISTRY["so100"].model_path, REGISTRY["so101"].model_path
    option = re.search(r"<option[^>]*/>", so101.read_text()).group(0)
    xml = re.sub(r"<option[^>]*/>", option, so100.read_text(), count=1)
    (tmp / "assets").symlink_to(so100.parent / "assets")
    out = tmp / so100.name
    out.write_text(xml)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lock", default="svla_so100_pickplace")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "so100_model_choice.json")
    args = ap.parse_args()
    lock = RobotLock.load(ROOT / "locks" / f"{args.lock}.lock.json")
    ds = load_dataset(lock.dataset)
    by = {e.index: e for e in ds.episodes}
    train = [by[i] for i in lock.result.train_episodes]
    eval_ = [by[i] for i in lock.result.eval_episodes]
    with tempfile.TemporaryDirectory() as tmp:
        register(variant(REGISTRY["so100"], "so100_deg", **DEGREES))
        register(variant(REGISTRY["so100"], "so100_so101_options_deg",
                         model_path=so100_with_so101_options(Path(tmp)), **DEGREES))
        out = {"lock": args.lock, "dataset": lock.dataset, "models": {}}
        for name in ("so101_deg", "so100_deg", "so100_so101_options_deg"):
            sim = default_simulate(robot=name)
            res = calibrate(train, eval_, simulate=sim, max_evaluations=150, seed=0)
            row = {
                "default_eval_deg": float(score(SimParams(), eval_, sim).overall_deg),
                "lock_eval_deg": float(score(lock.params, eval_, sim).overall_deg),
                "fresh_eval_deg": float(res.eval_score),
                "fresh_params": res.params.to_dict(),
            }
            out["models"][name] = row
            print(f"{name}: default {row['default_eval_deg']:.3f}  lock {row['lock_eval_deg']:.3f}  "
                  f"fresh {row['fresh_eval_deg']:.3f} deg", flush=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
