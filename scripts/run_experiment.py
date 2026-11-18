"""Train + group-CV + score ONE model from experiments/zoo/.
   python scripts/run_experiment.py lightgbm_gbdt
   python scripts/run_experiment.py knn --params '{"n_neighbors": 25}' --tag knn_k25   (optional override)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.harness import run_experiment

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--params", default=None, help="JSON dict overriding PARAMS")
    ap.add_argument("--tag", default=None, help="output name (default: model name)")
    a = ap.parse_args()
    run_experiment(a.name, json.loads(a.params) if a.params else None, a.tag)
