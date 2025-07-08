"""
Agentic MedTagger - A Python-native clinical NLP framework with agentic capabilities.

This package provides a comprehensive suite of tools for clinical text processing,
including fast dictionary matching, rule-based processing, and LLM-powered
normalization with active learning capabilities.
"""

__version__ = "0.1.0"
__author__ = "Sonish Sivarajkumar"
__email__ = "sonishsivarajkumar@example.com"

from .core.pipeline import Pipeline
from .core.document import Document
from .core.annotation import Annotation

__all__ = ["Pipeline", "Document", "Annotation"]
