"""
Document class for representing clinical documents.
"""

import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

from .annotation import AnnotationCollection, Annotation


@dataclass
class DocumentMetadata:
    """Metadata for clinical documents."""
    
    # Document identification
    document_id: str
    source_file: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Clinical metadata
    patient_id: Optional[str] = None
    encounter_id: Optional[str] = None
    document_type: Optional[str] = None  # 'progress_note', 'discharge_summary', etc.
    specialty: Optional[str] = None
    
    # Processing metadata
    processed_at: Optional[datetime] = None
    processor_version: Optional[str] = None
    
    # Custom attributes
    custom_attributes: Dict[str, Any] = field(default_factory=dict)


class Document:
    """
    Represents a clinical document with text content and annotations.
    
    The Document class serves as the central data structure for clinical
    text processing, containing the original text, metadata, and all
    annotations produced by the processing pipeline.
    """
    
    def __init__(
        self,
        text: str,
        metadata: Optional[DocumentMetadata] = None,
        document_id: Optional[str] = None
    ):
        self.text = text
        self.document_id = document_id or str(uuid.uuid4())
        
        # Initialize metadata
        if metadata is None:
            metadata = DocumentMetadata(document_id=self.document_id)
        self.metadata = metadata
        
        # Initialize annotations
        self.annotations = AnnotationCollection()
        
        # Processing state
        self._processed_sections = set()
        self._processing_errors = []
    
    @classmethod
    def from_file(cls, file_path: str) -> "Document":
        """Create document from a text file."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read content based on file type
        if path.suffix.lower() == '.txt':
            text = path.read_text(encoding='utf-8')
        elif path.suffix.lower() == '.pdf':
            # TODO: Implement PDF reading
            raise NotImplementedError("PDF reading not yet implemented")
        elif path.suffix.lower() in ['.doc', '.docx']:
            # TODO: Implement Word document reading
            raise NotImplementedError("Word document reading not yet implemented")
        else:
            # Default to text
            text = path.read_text(encoding='utf-8')
        
        metadata = DocumentMetadata(
            document_id=str(uuid.uuid4()),
            source_file=str(path.absolute())
        )
        
        return cls(text=text, metadata=metadata)
    
    def add_annotation(self, annotation: Annotation) -> None:
        """Add an annotation to the document."""
        # Validate span is within document bounds
        if annotation.span.end > len(self.text):
            raise ValueError(
                f"Annotation span {annotation.span.end} exceeds document length {len(self.text)}"
            )
        
        # Validate span text matches document text
        actual_text = self.text[annotation.span.start:annotation.span.end]
        if actual_text != annotation.span.text:
            raise ValueError(
                f"Annotation span text '{annotation.span.text}' does not match "
                f"document text '{actual_text}'"
            )
        
        self.annotations.add(annotation)
    
    def get_annotations_by_type(self, annotation_type: str) -> List[Annotation]:
        """Get all annotations of a specific type."""
        return self.annotations.filter_by_type(annotation_type)
    
    def get_annotations_by_label(self, label: str) -> List[Annotation]:
        """Get all annotations with a specific label."""
        return self.annotations.filter_by_label(label)
    
    def get_annotations_in_span(self, start: int, end: int) -> List[Annotation]:
        """Get all annotations that overlap with the given span."""
        overlapping = []
        for ann in self.annotations:
            # Check for overlap
            if not (ann.span.end <= start or ann.span.start >= end):
                overlapping.append(ann)
        return overlapping
    
    def get_text_with_highlights(self, annotation_types: List[str] = None) -> str:
        """
        Get document text with annotations highlighted.
        
        Args:
            annotation_types: List of annotation types to highlight.
                            If None, highlights all annotations.
        """
        if annotation_types is None:
            annotations_to_highlight = list(self.annotations)
        else:
            annotations_to_highlight = []
            for ann_type in annotation_types:
                annotations_to_highlight.extend(
                    self.annotations.filter_by_type(ann_type)
                )
        
        # Sort annotations by position (reverse order for replacement)
        sorted_annotations = sorted(
            annotations_to_highlight,
            key=lambda x: x.span.start,
            reverse=True
        )
        
        # Apply highlights
        highlighted_text = self.text
        for ann in sorted_annotations:
            start, end = ann.span.start, ann.span.end
            original = highlighted_text[start:end]
            highlighted = f"[{original}|{ann.label}]"
            highlighted_text = highlighted_text[:start] + highlighted + highlighted_text[end:]
        
        return highlighted_text
    
    def mark_section_processed(self, section_name: str) -> None:
        """Mark a processing section as completed."""
        self._processed_sections.add(section_name)
    
    def is_section_processed(self, section_name: str) -> bool:
        """Check if a processing section has been completed."""
        return section_name in self._processed_sections
    
    def add_processing_error(self, error: str, section: str = None) -> None:
        """Add a processing error."""
        error_info = {
            "error": error,
            "section": section,
            "timestamp": datetime.utcnow()
        }
        self._processing_errors.append(error_info)
    
    def get_processing_errors(self) -> List[Dict[str, Any]]:
        """Get all processing errors."""
        return self._processing_errors.copy()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary format."""
        return {
            "document_id": self.document_id,
            "text": self.text,
            "metadata": {
                "document_id": self.metadata.document_id,
                "source_file": self.metadata.source_file,
                "created_at": self.metadata.created_at.isoformat(),
                "patient_id": self.metadata.patient_id,
                "encounter_id": self.metadata.encounter_id,
                "document_type": self.metadata.document_type,
                "specialty": self.metadata.specialty,
                "processed_at": self.metadata.processed_at.isoformat() if self.metadata.processed_at else None,
                "processor_version": self.metadata.processor_version,
                "custom_attributes": self.metadata.custom_attributes
            },
            "annotations": self.annotations.to_json_format(),
            "processing_state": {
                "processed_sections": list(self._processed_sections),
                "processing_errors": self._processing_errors
            }
        }
    
    def to_xmi(self) -> str:
        """Export document to XMI format (UIMA compatible)."""
        # TODO: Implement XMI export
        raise NotImplementedError("XMI export not yet implemented")
    
    def __len__(self) -> int:
        """Return the length of the document text."""
        return len(self.text)
    
    def __str__(self) -> str:
        """Return a string representation of the document."""
        return f"Document({self.document_id}, {len(self.text)} chars, {len(self.annotations)} annotations)"
