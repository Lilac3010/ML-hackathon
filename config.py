"""Central configuration for PathSnap. Every team member imports from here.

Raw data location: set env var PATHSNAP_DATA to the folder that contains
`train/` and `student_test/` (each holding one sub-folder per trip). If unset we
fall back to the layout used on Person 1's machine.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

_default_data = ROOT.parent / "Student folder" / "Student folder"
DATA_DIR = Path(os.environ.get("PATHSNAP_DATA", _default_data))

PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

SEED = 42

# ---- candidate generation (tuned in scripts/candidate_study.py) -------------
CAND_RADIUS_M = 50.0       # first search radius around each GPS point
CAND_RADIUS_MAX_M = 400.0  # radius is doubled up to this if a point has no candidate
CAND_MAX_K = 20            # keep at most K nearest directed arcs per GPS point

# ---- reference alignment (route -> per-point reference arc) -----------------
REF_CAP_M = 60.0           # cap on point->arc distance in the alignment cost
REF_VALID_M = 40.0         # a point's reference arc is "valid" if within this distance

PARTITIONS = ("train", "student_test")
