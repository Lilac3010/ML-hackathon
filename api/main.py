"""PathSnap FastAPI backend (Person 3). Runs on Laptop 1.

Start:  uvicorn api.main:app --host 0.0.0.0 --port 8000
Env vars (all optional):
  PATHSNAP_DATA       folder containing train/ and student_test/ (used by config.py)
  PATHSNAP_TRIP_DIR   folder with one sub-folder per trip to serve (default: <data>/student_test;
                      on the evaluation day point it at the evaluator trips)
  PATHSNAP_MODEL      model name in models/<name>.joblib (default: lightgbm_gbdt)

Every request runs the full pipeline (features -> model -> HMM). Nothing is cached or hard-coded.
The reference route is read server-side inside run_trip(evaluate=True) and is never returned.
"""
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
from pathsnap.inference import load_hmm_params, run_trip

MODEL_NAME = os.environ.get("PATHSNAP_MODEL", "lightgbm_gbdt")
TRIP_DIR = Path(os.environ.get("PATHSNAP_TRIP_DIR", str(config.DATA_DIR / "student_test")))

LOG_DIR = config.ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_DIR / "api.log", encoding="utf-8")],
)
log = logging.getLogger("pathsnap.api")

state = {}


@asynccontextmanager
async def lifespan(app):
    art = joblib.load(config.ROOT / "models" / f"{MODEL_NAME}.joblib")
    state["model"] = art["model"]
    state["cols"] = art["feature_cols"]
    state["hmm"] = load_hmm_params()
    log.info("STARTUP model=%s trip_dir=%s trips=%d hmm_params=%s",
             MODEL_NAME, TRIP_DIR, len(trip_ids()), state["hmm"])
    yield


app = FastAPI(title="PathSnap API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def trip_ids():
    if not TRIP_DIR.is_dir():
        return []
    return sorted(p.name for p in TRIP_DIR.iterdir()
                  if p.is_dir() and (p / f"{p.name}.track").exists())


def n_points(tid):
    with open(TRIP_DIR / tid / f"{tid}.track") as f:
        return sum(1 for _ in f)


def predict_proba(X):
    return state["model"].predict_proba(X[state["cols"]])[:, 1]


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "model_loaded": "model" in state,
            "trip_dir": str(TRIP_DIR), "n_trips": len(trip_ids())}


@app.get("/trips")
def trips():
    ids = trip_ids()
    log.info("GET /trips -> %d trips", len(ids))
    return {"trips": [{"trip_id": t, "n_points": n_points(t)} for t in ids]}


@app.post("/predict")
def predict(req: PredictRequest):
    tid = req.trip_id.strip()
    if tid not in trip_ids():
        log.warning("POST /predict unknown trip_id=%r", tid)
        raise HTTPException(status_code=404, detail="unknown trip_id")
    has_ref = (TRIP_DIR / tid / f"{tid}.route").exists()
    t0 = time.perf_counter()
    try:
        out = run_trip(TRIP_DIR / tid, predict_proba, state["hmm"], evaluate=has_ref)
    except Exception as e:
        log.exception("POST /predict failed trip_id=%s", tid)
        raise HTTPException(status_code=500, detail=f"prediction failed: {e}")
    out["model"] = MODEL_NAME
    out["api_time_s"] = time.perf_counter() - t0
    m = out.get("metrics", {})
    log.info("POST /predict trip_id=%s n_obs=%d hmm_top1=%s clf_top1=%s total=%.2fs",
             tid, out["n_obs"],
             m.get("hmm", {}).get("top1_accuracy"), m.get("classifier_only", {}).get("top1_accuracy"),
             out["api_time_s"])
    return out
