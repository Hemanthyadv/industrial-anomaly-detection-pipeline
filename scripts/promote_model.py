"""Promote a model version to Production stage."""
import sys
sys.path.insert(0, ".")
import argparse
from src.ml.mlflow_setup import transition_to_stage

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="industrial-anomaly-detector")
    p.add_argument("--version", type=int, required=True)
    p.add_argument("--stage", default="Production")
    args = p.parse_args()
    mv = transition_to_stage(args.name, args.version, args.stage)
    print(f"Transitioned {args.name} v{args.version} to {args.stage}")
