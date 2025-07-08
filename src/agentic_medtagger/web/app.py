"""
FastAPI web application for Agentic MedTagger.
"""

import logging
from typing import List, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..core.pipeline import Pipeline
from ..core.document import Document, DocumentMetadata


logger = logging.getLogger(__name__)


# Pydantic models for API
class DocumentResponse(BaseModel):
    """Response model for document data."""
    document_id: str
    text: str
    metadata: Dict[str, Any]
    annotations: List[Dict[str, Any]]
    processing_state: Dict[str, Any]


class ProcessingRequest(BaseModel):
    """Request model for document processing."""
    text: str
    document_type: str = None
    patient_id: str = None
    encounter_id: str = None


class AnnotationFeedback(BaseModel):
    """Model for annotation feedback."""
    annotation_id: str
    feedback_type: str  # 'correct', 'incorrect', 'partial'
    corrected_label: str = None
    comments: str = None


# Global pipeline instance
pipeline = None


async def get_pipeline() -> Pipeline:
    """Get or create the global pipeline instance."""
    global pipeline
    if pipeline is None:
        pipeline = Pipeline()
        # Add default components
        from ..core.pipeline import TokenizerComponent, DictionaryMatcherComponent
        pipeline.add_component(TokenizerComponent())
        pipeline.add_component(DictionaryMatcherComponent())
        await pipeline.initialize()
    return pipeline


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="Agentic MedTagger",
        description="A Python-native clinical NLP framework with agentic capabilities",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc"
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "service": "agentic-medtagger"}
    
    # Document processing endpoints
    @app.post("/api/documents/process", response_model=DocumentResponse)
    async def process_document(
        request: ProcessingRequest,
        pipeline: Pipeline = Depends(get_pipeline)
    ):
        """Process a text document through the NLP pipeline."""
        try:
            # Create document
            metadata = DocumentMetadata(
                document_id="",  # Will be auto-generated
                document_type=request.document_type,
                patient_id=request.patient_id,
                encounter_id=request.encounter_id
            )
            
            document = Document(text=request.text, metadata=metadata)
            
            # Process document
            processed_doc = await pipeline.process_document(document)
            
            # Return response
            return DocumentResponse(**processed_doc.to_dict())
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/documents/upload")
    async def upload_document(
        file: UploadFile = File(...),
        pipeline: Pipeline = Depends(get_pipeline)
    ):
        """Upload and process a document file."""
        try:
            # Read file content
            content = await file.read()
            text = content.decode('utf-8')
            
            # Create document
            metadata = DocumentMetadata(
                document_id="",
                source_file=file.filename
            )
            
            document = Document(text=text, metadata=metadata)
            
            # Process document
            processed_doc = await pipeline.process_document(document)
            
            return DocumentResponse(**processed_doc.to_dict())
            
        except Exception as e:
            logger.error(f"File upload processing failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/documents/{document_id}")
    async def get_document(document_id: str):
        """Retrieve a processed document by ID."""
        # TODO: Implement document retrieval from database
        raise HTTPException(status_code=501, detail="Not implemented yet")
    
    @app.get("/api/documents/{document_id}/annotations")
    async def get_document_annotations(document_id: str):
        """Get annotations for a specific document."""
        # TODO: Implement annotation retrieval
        raise HTTPException(status_code=501, detail="Not implemented yet")
    
    # Annotation feedback endpoints
    @app.post("/api/annotations/feedback")
    async def submit_annotation_feedback(feedback: AnnotationFeedback):
        """Submit feedback for an annotation (active learning)."""
        try:
            # TODO: Implement feedback storage and processing
            logger.info(f"Received feedback for annotation {feedback.annotation_id}")
            return {"status": "feedback_recorded", "annotation_id": feedback.annotation_id}
            
        except Exception as e:
            logger.error(f"Feedback submission failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Pipeline management endpoints
    @app.get("/api/pipeline/status")
    async def get_pipeline_status(pipeline: Pipeline = Depends(get_pipeline)):
        """Get pipeline status and statistics."""
        try:
            stats = pipeline.get_statistics()
            return {
                "pipeline_name": pipeline.name,
                "version": pipeline.version,
                "initialized": pipeline.initialized,
                "components": [comp.name for comp in pipeline.components],
                "statistics": stats
            }
        except Exception as e:
            logger.error(f"Failed to get pipeline status: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/pipeline/components")
    async def list_pipeline_components(pipeline: Pipeline = Depends(get_pipeline)):
        """List all pipeline components and their configurations."""
        components = []
        for comp in pipeline.components:
            components.append({
                "name": comp.name,
                "enabled": comp.enabled,
                "config": comp.config
            })
        return {"components": components}
    
    # Configuration endpoints
    @app.get("/api/config/dictionaries")
    async def list_dictionaries():
        """List available dictionaries."""
        # TODO: Implement dictionary listing
        return {"dictionaries": []}
    
    @app.get("/api/config/models")
    async def list_models():
        """List available models."""
        # TODO: Implement model listing
        return {"models": []}
    
    # Static files for frontend
    frontend_path = Path(__file__).parent.parent.parent.parent / "frontend" / "build"
    if frontend_path.exists():
        app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="static")
    
    return app


# Create the app instance
app = create_app()
