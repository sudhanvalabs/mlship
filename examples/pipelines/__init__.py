"""Example pipelines for ShipML."""

from .sentiment_pipeline import SentimentPipeline
from .sklearn_normalization import NormalizationPipeline

__all__ = ["SentimentPipeline", "NormalizationPipeline"]
