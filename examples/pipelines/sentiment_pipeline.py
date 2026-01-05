"""Example pipeline for HuggingFace sentiment analysis model.

This shows how to create a custom pipeline for preprocessing and postprocessing.

Usage:
    python examples/huggingface_example.py  # Creates sentiment-model/
    shipml serve sentiment-model/ --pipeline examples.pipelines.sentiment_pipeline.SentimentPipeline

Test:
    curl -X POST http://localhost:8000/predict \
      -H "Content-Type: application/json" \
      -d '{"text": "This product is amazing!"}'
"""

from shipml.pipeline import Pipeline
import torch


class SentimentPipeline(Pipeline):
    """Custom pipeline for sentiment analysis with HuggingFace models."""

    def __init__(self, model_path):
        """Initialize with tokenizer from model directory."""
        super().__init__(model_path)

        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def preprocess(self, request_data):
        """
        Convert text input to tokenized tensors.

        Args:
            request_data: {"text": "This is great!"}

        Returns:
            Tokenized tensors ready for model
        """
        text = request_data.get("text", "")
        return self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    def postprocess(self, model_output):
        """
        Convert model output to human-readable sentiment.

        Args:
            model_output: Raw model output (logits)

        Returns:
            {"sentiment": "POSITIVE", "confidence": 0.99}
        """
        # Get probabilities from logits
        logits = model_output.logits if hasattr(model_output, "logits") else model_output
        probabilities = torch.softmax(logits, dim=-1)

        # Get prediction
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_class].item()

        # Map to sentiment labels (for binary classification)
        sentiments = {0: "NEGATIVE", 1: "POSITIVE"}
        sentiment = sentiments.get(predicted_class, f"CLASS_{predicted_class}")

        return {"sentiment": sentiment, "confidence": round(confidence, 4)}
