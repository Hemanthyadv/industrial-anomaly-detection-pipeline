"""CLI entry point for model training."""
import sys
sys.path.insert(0, ".")
from src.ml.train_models import train_all

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="models")
    p.add_argument("--data", default=None)
    args = p.parse_args()
    results = train_all(args.output_dir, args.data)
    for name, metrics in results.items():
        print(f"{name}: F1={metrics.get('f1', 'N/A'):.4f}")
