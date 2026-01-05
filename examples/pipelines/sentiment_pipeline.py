"""Example pipeline for HuggingFace sentiment analysis model.

This shows how to create a custom pipeline for preprocessing and postprocessing.

Usage:
    python examples/huggingface_example.py  # Creates sentiment-model/
    mlship serve sentiment-model/ --pipeline examples.pipelines.sentiment_pipeline.SentimentPipeline

Test:
    curl -X POST http://localhost:8000/predict \
      -H "Content-Type: application/json" \
      -d '{"text": "This product is amazing!"}'
"""

from mlship.pipeline import Pipeline


class SentimentPipeline(Pipeline):
    """Custom pipeline for sentiment analysis with HuggingFace models."""

    def preprocess(self, request_data):
        """
        Extract text from request.

        HuggingFace pipelines handle tokenization internally, so just return raw text.

        Args:
            request_data: {"text": "This is great!"}

        Returns:
            Raw text string
        """
        return request_data.get("text", "")

    def postprocess(self, model_output):
        """
        Convert pipeline output to custom format.

        Args:
            model_output: Pipeline output like [{"label": "POSITIVE", "score": 0.999}]

        Returns:
            {"sentiment": "POSITIVE", "confidence": 0.999}
        """
        # HuggingFace pipeline returns a list of predictions
        if isinstance(model_output, list):
            result = model_output[0]
        else:
            result = model_output

        return {
            "sentiment": result["label"],
            "confidence": round(result["score"], 4),
        }
