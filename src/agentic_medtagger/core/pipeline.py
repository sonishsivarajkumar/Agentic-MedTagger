"""
Core processing pipeline for Agentic MedTagger.
"""

import logging
from typing import List, Dict, Any, Optional, Type
from datetime import datetime
import asyncio

from .document import Document
from .annotation import Annotation


logger = logging.getLogger(__name__)


class PipelineComponent:
    """Base class for pipeline components."""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
    
    async def process(self, document: Document) -> Document:
        """Process a document and return the modified document."""
        raise NotImplementedError("Subclasses must implement process method")
    
    def validate_config(self) -> bool:
        """Validate component configuration."""
        return True


class TokenizerComponent(PipelineComponent):
    """spaCy-based tokenization component."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("tokenizer", config)
        self.model_name = self.config.get('spacy_model', 'en_core_web_sm')
        self.nlp = None
    
    async def initialize(self):
        """Initialize spaCy model."""
        try:
            import spacy
            self.nlp = spacy.load(self.model_name)
            logger.info(f"Loaded spaCy model: {self.model_name}")
        except OSError:
            logger.error(f"Failed to load spaCy model: {self.model_name}")
            raise
    
    async def process(self, document: Document) -> Document:
        """Tokenize document text using spaCy."""
        if not self.nlp:
            await self.initialize()
        
        doc = self.nlp(document.text)
        
        # Add token annotations
        for token in doc:
            annotation = Annotation(
                id=f"token_{token.i}",
                span={"start": token.idx, "end": token.idx + len(token.text), "text": token.text},
                label="TOKEN",
                annotation_type="token",
                annotator=self.name,
                attributes={
                    "pos": token.pos_,
                    "lemma": token.lemma_,
                    "is_alpha": token.is_alpha,
                    "is_stop": token.is_stop
                }
            )
            document.add_annotation(annotation)
        
        document.mark_section_processed("tokenization")
        return document


class DictionaryMatcherComponent(PipelineComponent):
    """flashtext-based dictionary matching component."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("dictionary_matcher", config)
        self.keyword_processor = None
        self.dictionaries = self.config.get('dictionaries', [])
    
    async def initialize(self):
        """Initialize flashtext keyword processor."""
        try:
            from flashtext import KeywordProcessor
            self.keyword_processor = KeywordProcessor()
            
            # Load dictionaries
            for dict_config in self.dictionaries:
                await self._load_dictionary(dict_config)
            
            logger.info(f"Loaded {len(self.dictionaries)} dictionaries")
        except ImportError:
            logger.error("flashtext library not installed")
            raise
    
    async def _load_dictionary(self, dict_config: Dict[str, Any]):
        """Load a single dictionary."""
        dict_type = dict_config.get('type', 'csv')
        file_path = dict_config.get('path')
        label = dict_config.get('label', 'ENTITY')
        
        if dict_type == 'csv':
            # TODO: Implement CSV loading
            pass
        elif dict_type == 'umls':
            # TODO: Implement UMLS loading
            pass
    
    async def process(self, document: Document) -> Document:
        """Find dictionary matches in document text."""
        if not self.keyword_processor:
            await self.initialize()
        
        # Find keywords
        keywords_found = self.keyword_processor.extract_keywords(
            document.text, span_info=True
        )
        
        # Create annotations
        for keyword, start_idx, end_idx in keywords_found:
            annotation = Annotation(
                id=f"dict_match_{start_idx}_{end_idx}",
                span={
                    "start": start_idx,
                    "end": end_idx,
                    "text": document.text[start_idx:end_idx]
                },
                label=keyword,
                annotation_type="dictionary_match",
                annotator=self.name
            )
            document.add_annotation(annotation)
        
        document.mark_section_processed("dictionary_matching")
        return document


class Pipeline:
    """
    Main processing pipeline for clinical NLP.
    
    The Pipeline orchestrates the execution of various components
    in a configurable sequence to process clinical documents.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.components: List[PipelineComponent] = []
        self.initialized = False
        
        # Pipeline metadata
        self.name = self.config.get('name', 'default_pipeline')
        self.version = self.config.get('version', '1.0.0')
        
        # Performance tracking
        self.processing_times = []
        self.error_count = 0
    
    def add_component(self, component: PipelineComponent) -> None:
        """Add a component to the pipeline."""
        if component.enabled:
            self.components.append(component)
            logger.info(f"Added component: {component.name}")
        else:
            logger.info(f"Skipped disabled component: {component.name}")
    
    def remove_component(self, component_name: str) -> bool:
        """Remove a component from the pipeline."""
        for i, component in enumerate(self.components):
            if component.name == component_name:
                del self.components[i]
                logger.info(f"Removed component: {component_name}")
                return True
        return False
    
    async def initialize(self) -> None:
        """Initialize all pipeline components."""
        logger.info(f"Initializing pipeline: {self.name}")
        
        for component in self.components:
            if hasattr(component, 'initialize'):
                try:
                    await component.initialize()
                    logger.info(f"Initialized component: {component.name}")
                except Exception as e:
                    logger.error(f"Failed to initialize component {component.name}: {e}")
                    raise
        
        self.initialized = True
        logger.info("Pipeline initialization complete")
    
    async def process_document(self, document: Document) -> Document:
        """Process a single document through the pipeline."""
        if not self.initialized:
            await self.initialize()
        
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Processing document: {document.document_id}")
            
            # Process through each component
            for component in self.components:
                try:
                    component_start = datetime.utcnow()
                    document = await component.process(document)
                    component_time = (datetime.utcnow() - component_start).total_seconds()
                    
                    logger.debug(
                        f"Component {component.name} processed document "
                        f"{document.document_id} in {component_time:.3f}s"
                    )
                    
                except Exception as e:
                    error_msg = f"Component {component.name} failed: {e}"
                    logger.error(error_msg)
                    document.add_processing_error(error_msg, component.name)
                    self.error_count += 1
                    
                    # Decide whether to continue or halt
                    if self.config.get('halt_on_error', False):
                        raise
            
            # Update document metadata
            document.metadata.processed_at = datetime.utcnow()
            document.metadata.processor_version = self.version
            
            # Track processing time
            total_time = (datetime.utcnow() - start_time).total_seconds()
            self.processing_times.append(total_time)
            
            logger.info(
                f"Document {document.document_id} processed successfully "
                f"in {total_time:.3f}s with {len(document.annotations)} annotations"
            )
            
            return document
            
        except Exception as e:
            logger.error(f"Pipeline failed for document {document.document_id}: {e}")
            raise
    
    async def process_documents(self, documents: List[Document]) -> List[Document]:
        """Process multiple documents through the pipeline."""
        results = []
        
        # Process documents concurrently if configured
        max_concurrent = self.config.get('max_concurrent_documents', 1)
        
        if max_concurrent > 1:
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def process_with_semaphore(doc):
                async with semaphore:
                    return await self.process_document(doc)
            
            tasks = [process_with_semaphore(doc) for doc in documents]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Document {documents[i].document_id} failed: {result}")
                    documents[i].add_processing_error(str(result))
                    processed_results.append(documents[i])
                else:
                    processed_results.append(result)
            
            return processed_results
        else:
            # Sequential processing
            for document in documents:
                try:
                    processed_doc = await self.process_document(document)
                    results.append(processed_doc)
                except Exception as e:
                    logger.error(f"Document {document.document_id} failed: {e}")
                    document.add_processing_error(str(e))
                    results.append(document)
            
            return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline processing statistics."""
        if not self.processing_times:
            return {"message": "No documents processed yet"}
        
        return {
            "documents_processed": len(self.processing_times),
            "total_processing_time": sum(self.processing_times),
            "average_processing_time": sum(self.processing_times) / len(self.processing_times),
            "min_processing_time": min(self.processing_times),
            "max_processing_time": max(self.processing_times),
            "error_count": self.error_count,
            "error_rate": self.error_count / len(self.processing_times) if self.processing_times else 0
        }
    
    @classmethod
    def from_config(cls, config_path: str) -> "Pipeline":
        """Create pipeline from configuration file."""
        # TODO: Implement configuration file loading
        raise NotImplementedError("Configuration file loading not yet implemented")
    
    def __str__(self) -> str:
        """Return a string representation of the pipeline."""
        component_names = [comp.name for comp in self.components]
        return f"Pipeline({self.name}, components: {component_names})"
