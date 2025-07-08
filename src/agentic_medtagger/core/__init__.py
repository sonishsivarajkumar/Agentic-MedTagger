"""Core module for Agentic MedTagger."""

from .annotation import Annotation
from .document import Document
from .pipeline import Pipeline

__all__ = ["Annotation", "Document", "Pipeline"]
