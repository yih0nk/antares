"""HTTP API behind the web UI. Run with ``antares serve``."""

from __future__ import annotations

from fastapi import FastAPI

import antares

# Datasets the demo offers out of the box.
DEMO_DATASETS = [
    "lerobot/svla_so101_pickplace",
    "lerobot/svla_so100_pickplace",
]

app = FastAPI(title="antares")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "version": antares.__version__}


@app.get("/api/datasets")
def datasets() -> list[str]:
    return DEMO_DATASETS
