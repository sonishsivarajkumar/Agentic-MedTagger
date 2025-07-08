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
    
    async def initialize(self):
        """Initialize the component."""
        pass
    
    async def process(self, document: Document) -> Document:
        """Process a document and return the modified document."""
        raise NotImplementedError("Subclasses must implement process method")
    
    def validate_config(self) -> bool:
        """Validate component configuration."""
        return True


class MedTaggerPipeline:
    """
    Main processing pipeline incorporating all MedTagger features.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.components = []
        self.initialized = False
        
        # Pipeline metrics
        self.metrics = {
            'documents_processed': 0,
            'total_processing_time': 0.0,
            'component_times': {},
            'errors': []
        }
    
    def add_component(self, component: PipelineComponent):
        """Add a component to the pipeline."""
        self.components.append(component)
        logger.info(f"Added component: {component.name}")
    
    async def initialize(self):
        """Initialize all pipeline components."""
        logger.info("Initializing MedTagger pipeline...")
        
        for component in self.components:
            if component.enabled:
                try:
                    await component.initialize()
                    logger.info(f"Initialized component: {component.name}")
                except Exception as e:
                    logger.error(f"Failed to initialize {component.name}: {e}")
                    raise
        
        self.initialized = True
        logger.info("Pipeline initialization complete")
    
    async def process_document(self, document: Document) -> Document:
        """Process a single document through the pipeline."""
        if not self.initialized:
            await self.initialize()
        
        start_time = datetime.now()
        
        try:
            # Process through each component
            for component in self.components:
                if component.enabled:
                    comp_start = datetime.now()
                    document = await component.process(document)
                    comp_time = (datetime.now() - comp_start).total_seconds()
                    
                    # Track component processing time
                    if component.name not in self.metrics['component_times']:
                        self.metrics['component_times'][component.name] = []
                    self.metrics['component_times'][component.name].append(comp_time)
            
            # Update metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self.metrics['documents_processed'] += 1
            self.metrics['total_processing_time'] += processing_time
            
            logger.debug(f"Processed document {document.document_id} in {processing_time:.3f}s")
            
        except Exception as e:
            error_info = {
                'document_id': document.document_id,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            self.metrics['errors'].append(error_info)
            logger.error(f"Error processing document {document.document_id}: {e}")
            raise
        
        return document
    
    async def process_documents(self, documents: List[Document]) -> List[Document]:
        """Process multiple documents."""
        processed_documents = []
        
        for document in documents:
            try:
                processed_doc = await self.process_document(document)
                processed_documents.append(processed_doc)
            except Exception as e:
                logger.error(f"Failed to process document {document.document_id}: {e}")
                # Add original document with error metadata
                if hasattr(document.metadata, '__dict__'):
                    document.metadata.processing_error = str(e)
                else:
                    document.metadata['processing_error'] = str(e)
                processed_documents.append(document)
        
        return processed_documents
    
    def get_component(self, name: str) -> Optional[PipelineComponent]:
        """Get a component by name."""
        for component in self.components:
            if component.name == name:
                return component
        return None
    
    def remove_component(self, name: str) -> bool:
        """Remove a component by name."""
        for i, component in enumerate(self.components):
            if component.name == name:
                del self.components[i]
                logger.info(f"Removed component: {name}")
                return True
        return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline processing metrics."""
        avg_processing_time = (
            self.metrics['total_processing_time'] / self.metrics['documents_processed']
            if self.metrics['documents_processed'] > 0 else 0
        )
        
        component_avg_times = {}
        for comp_name, times in self.metrics['component_times'].items():
            component_avg_times[comp_name] = sum(times) / len(times) if times else 0
        
        return {
            'documents_processed': self.metrics['documents_processed'],
            'total_processing_time': self.metrics['total_processing_time'],
            'average_processing_time': avg_processing_time,
            'component_average_times': component_avg_times,
            'error_count': len(self.metrics['errors']),
            'errors': self.metrics['errors'][-10:]  # Last 10 errors
        }
    
    def reset_metrics(self):
        """Reset pipeline metrics."""
        self.metrics = {
            'documents_processed': 0,
            'total_processing_time': 0.0,
            'component_times': {},
            'errors': []
        }


def create_medtagger_pipeline(config: Dict[str, Any] = None) -> MedTaggerPipeline:
    """
    Create a complete MedTagger pipeline with all components.
    """
    from ..annotators import (
        DictionaryMatcher, MedTaggerIE, MedTaggerML, 
        SectionDetector, AssertionNegationDetector, OMOPUMLSNormalizer
    )
    
    config = config or {}
    pipeline = MedTaggerPipeline(config)
    
    # Add components in processing order
    
    # 1. Section Detection (first to identify document structure)
    if config.get('enable_section_detection', True):
        section_config = config.get('section_detector', {})
        pipeline.add_component(SectionDetector(section_config))
    
    # 2. Dictionary Matching (fast keyword extraction)
    if config.get('enable_dictionary_matching', True):
        dict_config = config.get('dictionary_matcher', {})
        pipeline.add_component(DictionaryMatcher(dict_config))
    
    # 3. Pattern-based Information Extraction
    if config.get('enable_pattern_extraction', True):
        ie_config = config.get('medtagger_ie', {})
        pipeline.add_component(MedTaggerIE(ie_config))
    
    # 4. ML-based Named Entity Recognition
    if config.get('enable_ml_ner', True):
        ml_config = config.get('medtagger_ml', {})
        pipeline.add_component(MedTaggerML(ml_config))
    
    # 5. Assertion and Negation Detection
    if config.get('enable_assertion_negation', True):
        assertion_config = config.get('assertion_negation', {})
        pipeline.add_component(AssertionNegationDetector(assertion_config))
    
    # 6. OMOP/UMLS Normalization (last to normalize all found entities)
    if config.get('enable_normalization', True):
        norm_config = config.get('omop_umls_normalizer', {})
        pipeline.add_component(OMOPUMLSNormalizer(norm_config))
    
    return pipeline


# Legacy components for backward compatibility
