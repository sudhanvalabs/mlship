"""Simple test pipeline without dependencies."""

from shipml.pipeline import Pipeline


class SimplePipeline(Pipeline):
    """Minimal pipeline for testing."""

    def __init__(self, model_path):
        super().__init__(model_path)
        print(f"Pipeline initialized with model_path: {model_path}")

    def preprocess(self, request_data):
        """Pass through preprocessing."""
        print(f"Preprocess called with: {request_data}")
        return request_data.get("features", [])

    def postprocess(self, model_output):
        """Pass through postprocessing."""
        print(f"Postprocess called with: {model_output}")
        return model_output
