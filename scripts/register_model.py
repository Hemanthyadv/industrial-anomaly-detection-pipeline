"""Register a trained model in MLflow."""
import sys
sys.path.insert(0, ".")
import argparse
from src.ml.mlflow_setup import register_model

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-uri", required=True, help="MLflow model URI (runs:/<id>/model)")
    p.add_argument("--name", default="industrial-anomaly-detector")
    args = p.parse_args()
    mv = register_model(args.model_uri, args.name)
    print(f"Registered: {mv.name} v{mv.version}")
