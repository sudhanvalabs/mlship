"""Example pipelines for ShipML.

Note: Pipeline classes are imported dynamically to avoid dependency issues.
Import the specific pipeline you need directly:

    from examples.pipelines.sentiment_pipeline import SentimentPipeline
    from examples.pipelines.sklearn_normalization import NormalizationPipeline
"""

# Don't import here to avoid forcing all dependencies to be installed
__all__ = ["sentiment_pipeline", "sklearn_normalization", "simple_test"]
