"""Load and test the production model."""
import sys
sys.path.insert(0, ".")
import numpy as np
from src.ml.mlflow_setup import load_production_model

if __name__ == "__main__":
    model = load_production_model(
        "industrial-anomaly-detector",
        fallback_path="models/isolation_forest.joblib",
    )
    if model is None:
        print("No model available")
        sys.exit(1)
    # Test prediction
    X_test = np.array([[70.0, 2.0, 0.05]])
    try:
        pred = model.predict(X_test)
        print(f"Test prediction: {pred}")
    except Exception as e:
        print(f"Prediction error: {e}")
