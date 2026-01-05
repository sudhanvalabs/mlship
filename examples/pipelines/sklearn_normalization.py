"""Example pipeline for sklearn model with feature normalization.

This shows how to add preprocessing (normalization) and postprocessing
for a sklearn model.

Usage:
    # First, save normalization parameters with your model
    python examples/sklearn_example.py

    # Serve with pipeline
    mlship serve fraud_detector.pkl --pipeline examples.pipelines.sklearn_normalization.NormalizationPipeline

Test:
    curl -X POST http://localhost:8000/predict \
      -H "Content-Type: application/json" \
      -d '{"features": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}'
"""

from mlship.pipeline import Pipeline
import numpy as np
import json
from pathlib import Path


class NormalizationPipeline(Pipeline):
    """Pipeline that normalizes features before prediction."""

    def __init__(self, model_path):
        """Initialize and load normalization parameters."""
        super().__init__(model_path)

        # Load normalization parameters (mean and std)
        # These should be saved alongside your model
        config_path = Path(model_path).parent / "normalization_params.json"

        if config_path.exists():
            with open(config_path) as f:
                params = json.load(f)
                self.mean = np.array(params["mean"])
                self.std = np.array(params["std"])
        else:
            # Default: no normalization
            print("Warning: normalization_params.json not found. Using identity transform.")
            self.mean = 0
            self.std = 1

    def preprocess(self, request_data):
        """
        Normalize features using saved mean and std.

        Args:
            request_data: {"features": [1.0, 2.0, ...]}

        Returns:
            Normalized features
        """
        features = np.array(request_data["features"])

        # Normalize: (x - mean) / std
        normalized = (features - self.mean) / self.std

        return normalized.tolist()

    def postprocess(self, model_output):
        """
        Add additional info to prediction output.

        Args:
            model_output: {"prediction": 0, "probability": 0.87}

        Returns:
            Enhanced output with preprocessing info
        """
        # Add metadata about preprocessing
        model_output["preprocessing"] = "normalized"
        model_output["normalization"] = {
            "mean": self.mean.tolist() if isinstance(self.mean, np.ndarray) else self.mean,
            "std": self.std.tolist() if isinstance(self.std, np.ndarray) else self.std,
        }

        return model_output
