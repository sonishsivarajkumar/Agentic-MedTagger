"""Core module for Agentic MedTagger."""

from .annotation import Annotation
from .document import Document
from .pipeline import MedTaggerPipeline, create_medtagger_pipeline

__all__ = ["Annotation", "Document", "MedTaggerPipeline", "create_medtagger_pipeline"]
