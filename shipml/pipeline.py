"""Base pipeline interface for custom preprocessing and postprocessing."""

from typing import Any, Dict, Union, List
from abc import ABC, abstractmethod


class Pipeline(ABC):
    """
    Base class for custom preprocessing and postprocessing pipelines.

    Users can subclass this to define custom transformations for their models.

    Example:
        >>> class SentimentPipeline(Pipeline):
        ...     def __init__(self, model_path):
        ...         from transformers import AutoTokenizer
        ...         self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        ...
        ...     def preprocess(self, request_data):
        ...         text = request_data["text"]
        ...         return self.tokenizer(text, return_tensors="pt")
        ...
        ...     def postprocess(self, model_output):
        ...         import torch
        ...         probs = torch.softmax(model_output.logits, dim=-1)
        ...         return {
        ...             "sentiment": "POSITIVE" if probs[0][1] > 0.5 else "NEGATIVE",
        ...             "confidence": float(probs[0][1])
        ...         }
    """

    def __init__(self, model_path: str):
        """
        Initialize pipeline with access to model directory.

        Args:
            model_path: Path to the model file or directory
        """
        self.model_path = model_path

    @abstractmethod
    def preprocess(self, request_data: Dict[str, Any]) -> Any:
        """
        Transform API request data into model input format.

        This method is called before the model prediction.

        Args:
            request_data: Raw request data from the API (dict)

        Returns:
            Preprocessed data ready for model input

        Example:
            For a text model:
            Input:  {"text": "This is great!"}
            Output: Tokenized tensors
        """
        pass

    @abstractmethod
    def postprocess(self, model_output: Any) -> Dict[str, Any]:
        """
        Transform model output into API response format.

        This method is called after the model prediction.

        Args:
            model_output: Raw output from the model

        Returns:
            Dict formatted for API response

        Example:
            Input:  Raw tensor [[0.1, 0.9]]
            Output: {"sentiment": "POSITIVE", "confidence": 0.9}
        """
        pass
