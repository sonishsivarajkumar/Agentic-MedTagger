"""
Basic tests for Agentic MedTagger core functionality.
"""

import pytest
from datetime import datetime

from agentic_medtagger.core.annotation import Annotation, Span, AnnotationCollection
from agentic_medtagger.core.document import Document, DocumentMetadata


class TestAnnotation:
    """Test annotation functionality."""
    
    def test_span_creation(self):
        """Test span creation and validation."""
        span = Span(start=0, end=5, text="hello")
        assert span.start == 0
        assert span.end == 5
        assert span.text == "hello"
    
    def test_span_validation(self):
        """Test span validation."""
        with pytest.raises(ValueError):
            Span(start=-1, end=5, text="hello")
        
        with pytest.raises(ValueError):
            Span(start=5, end=3, text="hello")
    
    def test_annotation_creation(self):
        """Test annotation creation."""
        span = Span(start=0, end=5, text="hello")
        annotation = Annotation(
            id="test_1",
            span=span,
            label="GREETING",
            annotation_type="entity"
        )
        
        assert annotation.id == "test_1"
        assert annotation.span == span
        assert annotation.label == "GREETING"
        assert annotation.annotation_type == "entity"
        assert annotation.confidence == 1.0
        assert isinstance(annotation.timestamp, datetime)
    
    def test_annotation_to_dict(self):
        """Test annotation serialization."""
        span = Span(start=0, end=5, text="hello")
        annotation = Annotation(
            id="test_1",
            span=span,
            label="GREETING",
            annotation_type="entity",
            confidence=0.9
        )
        
        data = annotation.to_dict()
        assert data["id"] == "test_1"
        assert data["span"]["start"] == 0
        assert data["span"]["end"] == 5
        assert data["span"]["text"] == "hello"
        assert data["label"] == "GREETING"
        assert data["confidence"] == 0.9
    
    def test_annotation_from_dict(self):
        """Test annotation deserialization."""
        data = {
            "id": "test_1",
            "span": {"start": 0, "end": 5, "text": "hello"},
            "label": "GREETING",
            "annotation_type": "entity",
            "confidence": 0.9,
            "annotator": "test",
            "timestamp": "2025-01-01T00:00:00"
        }
        
        annotation = Annotation.from_dict(data)
        assert annotation.id == "test_1"
        assert annotation.span.start == 0
        assert annotation.label == "GREETING"
        assert annotation.confidence == 0.9


class TestAnnotationCollection:
    """Test annotation collection functionality."""
    
    def test_collection_creation(self):
        """Test collection creation."""
        collection = AnnotationCollection()
        assert len(collection) == 0
    
    def test_add_annotation(self):
        """Test adding annotations."""
        collection = AnnotationCollection()
        span = Span(start=0, end=5, text="hello")
        annotation = Annotation(
            id="test_1",
            span=span,
            label="GREETING",
            annotation_type="entity"
        )
        
        collection.add(annotation)
        assert len(collection) == 1
        assert collection[0] == annotation
    
    def test_filter_by_type(self):
        """Test filtering by annotation type."""
        collection = AnnotationCollection()
        
        # Add different types of annotations
        entities = [
            Annotation(f"ent_{i}", Span(i*5, i*5+3, "ent"), "ENTITY", "entity")
            for i in range(3)
        ]
        tokens = [
            Annotation(f"tok_{i}", Span(i*2, i*2+1, "t"), "TOKEN", "token")
            for i in range(2)
        ]
        
        for ann in entities + tokens:
            collection.add(ann)
        
        entity_annotations = collection.filter_by_type("entity")
        token_annotations = collection.filter_by_type("token")
        
        assert len(entity_annotations) == 3
        assert len(token_annotations) == 2


class TestDocument:
    """Test document functionality."""
    
    def test_document_creation(self):
        """Test document creation."""
        text = "This is a test document."
        doc = Document(text=text)
        
        assert doc.text == text
        assert len(doc.annotations) == 0
        assert doc.document_id is not None
        assert isinstance(doc.metadata, DocumentMetadata)
    
    def test_document_with_metadata(self):
        """Test document creation with metadata."""
        text = "This is a test document."
        metadata = DocumentMetadata(
            document_id="test_doc_1",
            document_type="progress_note",
            patient_id="PAT123"
        )
        
        doc = Document(text=text, metadata=metadata)
        assert doc.metadata.document_type == "progress_note"
        assert doc.metadata.patient_id == "PAT123"
    
    def test_add_annotation(self):
        """Test adding annotations to document."""
        text = "This is a test document."
        doc = Document(text=text)
        
        span = Span(start=0, end=4, text="This")
        annotation = Annotation(
            id="test_1",
            span=span,
            label="DEMONSTRATIVE",
            annotation_type="entity"
        )
        
        doc.add_annotation(annotation)
        assert len(doc.annotations) == 1
        assert doc.annotations[0] == annotation
    
    def test_annotation_validation(self):
        """Test annotation validation against document text."""
        text = "This is a test document."
        doc = Document(text=text)
        
        # Test span exceeds document
        with pytest.raises(ValueError):
            span = Span(start=0, end=100, text="invalid")
            annotation = Annotation("test_1", span, "LABEL", "entity")
            doc.add_annotation(annotation)
        
        # Test span text mismatch
        with pytest.raises(ValueError):
            span = Span(start=0, end=4, text="Wrong")
            annotation = Annotation("test_1", span, "LABEL", "entity")
            doc.add_annotation(annotation)
    
    def test_get_annotations_by_type(self):
        """Test retrieving annotations by type."""
        text = "This is a test document."
        doc = Document(text=text)
        
        # Add different types
        entity_ann = Annotation(
            "ent_1", Span(0, 4, "This"), "DEMONSTRATIVE", "entity"
        )
        token_ann = Annotation(
            "tok_1", Span(5, 7, "is"), "TOKEN", "token"
        )
        
        doc.add_annotation(entity_ann)
        doc.add_annotation(token_ann)
        
        entities = doc.get_annotations_by_type("entity")
        tokens = doc.get_annotations_by_type("token")
        
        assert len(entities) == 1
        assert len(tokens) == 1
        assert entities[0] == entity_ann
        assert tokens[0] == token_ann
    
    def test_document_serialization(self):
        """Test document serialization."""
        text = "This is a test document."
        doc = Document(text=text)
        
        # Add an annotation
        span = Span(start=0, end=4, text="This")
        annotation = Annotation("test_1", span, "DEMONSTRATIVE", "entity")
        doc.add_annotation(annotation)
        
        # Serialize
        data = doc.to_dict()
        
        assert data["text"] == text
        assert data["document_id"] == doc.document_id
        assert len(data["annotations"]) == 1
        assert data["annotations"][0]["id"] == "test_1"


if __name__ == "__main__":
    pytest.main([__file__])
