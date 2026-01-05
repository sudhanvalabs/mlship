"""Model loaders for different ML frameworks."""

from mlship.loaders.base import ModelLoader
from mlship.loaders.detector import detect_framework, get_loader

__all__ = ["ModelLoader", "detect_framework", "get_loader"]
