"""
Annotation classes for representing clinical NLP annotations.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from datetime import datetime


@dataclass
class Span:
    """Represents a text span with start and end positions."""
    start: int
    end: int
    text: str

    def __post_init__(self):
        if self.start < 0 or self.end < self.start:
            raise ValueError("Invalid span boundaries")


@dataclass
class Annotation:
    """
    Represents a clinical annotation with metadata.
    
    Supports various annotation types including named entities,
    sections, assertions, and normalized concepts.
    """
    
    # Core annotation data
    id: str
    span: Span
    label: str
    annotation_type: str  # 'entity', 'section', 'assertion', 'negation', etc.
    
    # Confidence and provenance
    confidence: float = 1.0
    annotator: str = "unknown"
    timestamp: datetime = None
    
    # Additional metadata
    attributes: Dict[str, Any] = None
    
    # Normalization data
    normalized_concept: Optional[str] = None
    umls_cui: Optional[str] = None
    umls_semantic_types: Optional[List[str]] = None
    
    # Relationship data
    relations: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.attributes is None:
            self.attributes = {}
        if self.relations is None:
            self.relations = []
        
        # Validate confidence
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
    
    @property
    def text(self) -> str:
        """Get the text content of the annotation."""
        return self.span.text
    
    @property 
    def start(self) -> int:
        """Get the start position of the annotation."""
        return self.span.start
    
    @property
    def end(self) -> int:
        """Get the end position of the annotation."""
        return self.span.end
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert annotation to dictionary format."""
        return {
            "id": self.id,
            "span": {
                "start": self.span.start,
                "end": self.span.end,
                "text": self.span.text
            },
            "label": self.label,
            "annotation_type": self.annotation_type,
            "confidence": self.confidence,
            "annotator": self.annotator,
            "timestamp": self.timestamp.isoformat(),
            "attributes": self.attributes,
            "normalized_concept": self.normalized_concept,
            "umls_cui": self.umls_cui,
            "umls_semantic_types": self.umls_semantic_types,
            "relations": self.relations
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Annotation":
        """Create annotation from dictionary."""
        span_data = data["span"]
        span = Span(
            start=span_data["start"],
            end=span_data["end"],
            text=span_data["text"]
        )
        
        timestamp = datetime.fromisoformat(data["timestamp"])
        
        return cls(
            id=data["id"],
            span=span,
            label=data["label"],
            annotation_type=data["annotation_type"],
            confidence=data.get("confidence", 1.0),
            annotator=data.get("annotator", "unknown"),
            timestamp=timestamp,
            attributes=data.get("attributes", {}),
            normalized_concept=data.get("normalized_concept"),
            umls_cui=data.get("umls_cui"),
            umls_semantic_types=data.get("umls_semantic_types"),
            relations=data.get("relations", [])
        )


class AnnotationCollection:
    """Collection of annotations with utility methods."""
    
    def __init__(self, annotations: List[Annotation] = None):
        self.annotations = annotations or []
    
    def add(self, annotation: Annotation) -> None:
        """Add an annotation to the collection."""
        self.annotations.append(annotation)
    
    def filter_by_type(self, annotation_type: str) -> List[Annotation]:
        """Filter annotations by type."""
        return [ann for ann in self.annotations if ann.annotation_type == annotation_type]
    
    def filter_by_label(self, label: str) -> List[Annotation]:
        """Filter annotations by label."""
        return [ann for ann in self.annotations if ann.label == label]
    
    def filter_by_confidence(self, min_confidence: float) -> List[Annotation]:
        """Filter annotations by minimum confidence."""
        return [ann for ann in self.annotations if ann.confidence >= min_confidence]
    
    def sort_by_position(self) -> List[Annotation]:
        """Sort annotations by text position."""
        return sorted(self.annotations, key=lambda x: (x.span.start, x.span.end))
    
    def to_json_format(self) -> List[Dict[str, Any]]:
        """Convert collection to JSON-serializable format."""
        return [ann.to_dict() for ann in self.annotations]
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __iter__(self):
        return iter(self.annotations)
    
    def __getitem__(self, index):
        return self.annotations[index]
