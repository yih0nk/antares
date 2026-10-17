"""``antares`` command line."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="antares")
    sub = parser.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="run the API server for the web UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    inspect = sub.add_parser("inspect", help="summarize a LeRobot dataset")
    inspect.add_argument("repo_id")

    args = parser.parse_args(argv)

    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("antares.server:app", host=args.host, port=args.port)
    elif args.cmd == "inspect":
        from antares.data import load_dataset

        ds = load_dataset(args.repo_id)
        frames = sum(len(e) for e in ds.episodes)
        print(f"{ds.repo_id}: {ds.robot_type}, {len(ds.episodes)} episodes, "
              f"{frames} frames @ {ds.fps:g} fps")
        print("joints:", ", ".join(ds.joint_names))
    return 0
