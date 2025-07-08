"""
MedTaggerML: Machine Learning-based Named Entity Recognition component.
Inspired by Mayo Clinic MedTagger's ML module.
"""

import logging
from typing import Dict, List, Any, Optional, Set, Tuple
import numpy as np
import spacy
from spacy.tokens import Doc, Span as SpacySpan
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch

from ..core.pipeline import PipelineComponent
from ..core.document import Document
from ..core.annotation import Annotation, Span


logger = logging.getLogger(__name__)


class MedTaggerML(PipelineComponent):
    """
    Machine Learning-based NER component using spaCy and transformers
    for clinical entity recognition.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("medtagger_ml", config)
        
        # Model configurations
        self.spacy_model = self.config.get('spacy_model', 'en_core_web_sm')
        self.biobert_model = self.config.get('biobert_model', 'dmis-lab/biobert-base-cased-v1.1')
        self.clinical_bert_model = self.config.get('clinical_bert_model', 'emilyalsentzer/Bio_ClinicalBERT')
        
        # Model instances
        self.nlp = None
        self.biobert_pipeline = None
        self.clinical_pipeline = None
        
        # NER configuration
        self.use_spacy = self.config.get('use_spacy', True)
        self.use_biobert = self.config.get('use_biobert', False)
        self.use_clinical_bert = self.config.get('use_clinical_bert', False)
        
        # Entity filtering
        self.entity_types = self.config.get('entity_types', [
            'PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE',
            'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL'
        ])
        
        # Clinical entity types
        self.clinical_entities = self.config.get('clinical_entities', [
            'DISEASE', 'SYMPTOM', 'MEDICATION', 'DOSAGE', 'TREATMENT', 'TEST',
            'ANATOMY', 'PROCEDURE', 'CONDITION'
        ])
        
        # Confidence thresholds
        self.min_confidence = self.config.get('min_confidence', 0.5)
        self.spacy_confidence = self.config.get('spacy_confidence', 0.8)
        
        # Aggregation settings
        self.aggregate_overlapping = self.config.get('aggregate_overlapping', True)
        self.prefer_longer_entities = self.config.get('prefer_longer_entities', True)
    
    async def initialize(self):
        """Initialize ML models."""
        logger.info(f"Initializing {self.name}")
        
        # Initialize spaCy model
        if self.use_spacy:
            await self._initialize_spacy()
        
        # Initialize BERT models
        if self.use_biobert:
            await self._initialize_biobert()
        
        if self.use_clinical_bert:
            await self._initialize_clinical_bert()
        
        logger.info(f"Initialized {self.name} with configured models")
    
    async def _initialize_spacy(self):
        """Initialize spaCy model."""
        try:
            self.nlp = spacy.load(self.spacy_model)
            logger.info(f"Loaded spaCy model: {self.spacy_model}")
        except OSError:
            logger.warning(f"Could not load spaCy model {self.spacy_model}, using blank model")
            self.nlp = spacy.blank("en")
    
    async def _initialize_biobert(self):
        """Initialize BioBERT model."""
        try:
            self.biobert_pipeline = pipeline(
                "ner",
                model=self.biobert_model,
                tokenizer=self.biobert_model,
                aggregation_strategy="simple"
            )
            logger.info(f"Loaded BioBERT model: {self.biobert_model}")
        except Exception as e:
            logger.error(f"Failed to load BioBERT model: {e}")
            self.use_biobert = False
    
    async def _initialize_clinical_bert(self):
        """Initialize Clinical BERT model."""
        try:
            self.clinical_pipeline = pipeline(
                "ner",
                model=self.clinical_bert_model,
                tokenizer=self.clinical_bert_model,
                aggregation_strategy="simple"
            )
            logger.info(f"Loaded Clinical BERT model: {self.clinical_bert_model}")
        except Exception as e:
            logger.error(f"Failed to load Clinical BERT model: {e}")
            self.use_clinical_bert = False
    
    async def process(self, document: Document) -> Document:
        """Process document with ML-based NER."""
        logger.debug(f"Processing document with {self.name}")
        
        # Extract entities using different models
        entities = []
        
        if self.use_spacy and self.nlp:
            spacy_entities = await self._extract_spacy_entities(document)
            entities.extend(spacy_entities)
        
        if self.use_biobert and self.biobert_pipeline:
            biobert_entities = await self._extract_bert_entities(
                document, self.biobert_pipeline, "BioBERT"
            )
            entities.extend(biobert_entities)
        
        if self.use_clinical_bert and self.clinical_pipeline:
            clinical_entities = await self._extract_bert_entities(
                document, self.clinical_pipeline, "ClinicalBERT"
            )
            entities.extend(clinical_entities)
        
        # Aggregate and filter entities
        final_entities = self._aggregate_entities(entities)
        
        # Add annotations to document
        for i, entity in enumerate(final_entities):
            annotation = Annotation(
                id=f"ml_{entity['metadata']['model']}_{entity['span'].start}_{entity['span'].end}_{i}",
                span=entity['span'],
                label=entity['label'],
                annotation_type='ml_entity',
                confidence=entity['confidence'],
                annotator=self.name,
                attributes=entity['metadata']
            )
            document.add_annotation(annotation)
        
        return document
    
    async def _extract_spacy_entities(self, document: Document) -> List[Dict[str, Any]]:
        """Extract entities using spaCy."""
        entities = []
        doc = self.nlp(document.text)
        
        for ent in doc.ents:
            if ent.label_ in self.entity_types:
                entity = {
                    'span': Span(ent.start_char, ent.end_char, ent.text),
                    'label': ent.label_,
                    'text': ent.text,
                    'confidence': self.spacy_confidence,
                    'metadata': {
                        'model': 'spaCy',
                        'model_name': self.spacy_model,
                        'entity_id': ent.ent_id_,
                        'kb_id': ent.kb_id_
                    }
                }
                entities.append(entity)
        
        return entities
    
    async def _extract_bert_entities(
        self, 
        document: Document, 
        pipeline_model, 
        model_name: str
    ) -> List[Dict[str, Any]]:
        """Extract entities using BERT-based models."""
        entities = []
        
        try:
            # Process text with BERT model
            results = pipeline_model(document.text)
            
            for result in results:
                confidence = result.get('score', 0.0)
                
                if confidence >= self.min_confidence:
                    # Map BERT entity labels to clinical entities
                    label = self._map_bert_label(result['entity_group'])
                    
                    entity = {
                        'span': Span(result['start'], result['end'], result['word']),
                        'label': label,
                        'text': result['word'],
                        'confidence': confidence,
                        'metadata': {
                            'model': model_name,
                            'original_label': result['entity_group'],
                            'bert_score': confidence
                        }
                    }
                    entities.append(entity)
        
        except Exception as e:
            logger.error(f"Error extracting entities with {model_name}: {e}")
        
        return entities
    
    def _map_bert_label(self, bert_label: str) -> str:
        """Map BERT entity labels to standardized clinical labels."""
        # Mapping from BERT labels to clinical labels
        label_mapping = {
            'B-PER': 'PERSON',
            'I-PER': 'PERSON',
            'B-ORG': 'ORGANIZATION',
            'I-ORG': 'ORGANIZATION',
            'B-LOC': 'LOCATION',
            'I-LOC': 'LOCATION',
            'B-MISC': 'MISCELLANEOUS',
            'I-MISC': 'MISCELLANEOUS',
            'B-DISEASE': 'DISEASE',
            'I-DISEASE': 'DISEASE',
            'B-SYMPTOM': 'SYMPTOM',
            'I-SYMPTOM': 'SYMPTOM',
            'B-MEDICATION': 'MEDICATION',
            'I-MEDICATION': 'MEDICATION',
            'B-DOSAGE': 'DOSAGE',
            'I-DOSAGE': 'DOSAGE',
            'B-TREATMENT': 'TREATMENT',
            'I-TREATMENT': 'TREATMENT',
            'B-TEST': 'TEST',
            'I-TEST': 'TEST',
            'B-ANATOMY': 'ANATOMY',
            'I-ANATOMY': 'ANATOMY',
            'B-PROCEDURE': 'PROCEDURE',
            'I-PROCEDURE': 'PROCEDURE'
        }
        
        return label_mapping.get(bert_label, bert_label.replace('B-', '').replace('I-', ''))
    
    def _aggregate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Aggregate overlapping entities and apply filtering."""
        if not self.aggregate_overlapping:
            return entities
        
        # Sort entities by start position
        entities.sort(key=lambda x: (x['span'].start, x['span'].end))
        
        aggregated = []
        
        for entity in entities:
            # Check for overlaps with existing entities
            overlapping_indices = []
            
            for i, existing in enumerate(aggregated):
                if self._entities_overlap(entity['span'], existing['span']):
                    overlapping_indices.append(i)
            
            if not overlapping_indices:
                # No overlap, add entity
                aggregated.append(entity)
            else:
                # Handle overlap
                if self.prefer_longer_entities:
                    # Keep the longer entity
                    longest_entity = entity
                    longest_length = entity['span'].end - entity['span'].start
                    
                    for i in overlapping_indices:
                        existing = aggregated[i]
                        existing_length = existing['span'].end - existing['span'].start
                        
                        if existing_length > longest_length:
                            longest_entity = existing
                            longest_length = existing_length
                    
                    # Remove overlapping entities and add the longest
                    for i in sorted(overlapping_indices, reverse=True):
                        del aggregated[i]
                    
                    aggregated.append(longest_entity)
                else:
                    # Keep the highest confidence entity
                    best_entity = entity
                    best_confidence = entity['confidence']
                    
                    for i in overlapping_indices:
                        existing = aggregated[i]
                        if existing['confidence'] > best_confidence:
                            best_entity = existing
                            best_confidence = existing['confidence']
                    
                    # Remove overlapping entities and add the best
                    for i in sorted(overlapping_indices, reverse=True):
                        del aggregated[i]
                    
                    aggregated.append(best_entity)
        
        return aggregated
    
    def _entities_overlap(self, span1: Span, span2: Span) -> bool:
        """Check if two entity spans overlap."""
        return not (span1.end <= span2.start or span2.end <= span1.start)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            'spacy_model': self.spacy_model if self.use_spacy else None,
            'biobert_model': self.biobert_model if self.use_biobert else None,
            'clinical_bert_model': self.clinical_bert_model if self.use_clinical_bert else None,
            'models_loaded': {
                'spacy': self.nlp is not None,
                'biobert': self.biobert_pipeline is not None,
                'clinical_bert': self.clinical_pipeline is not None
            }
        }
    
    async def evaluate_on_text(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Evaluate all models on a text and return results separately."""
        temp_doc = Document(text=text, id="temp")
        
        results = {}
        
        if self.use_spacy and self.nlp:
            results['spacy'] = await self._extract_spacy_entities(temp_doc)
        
        if self.use_biobert and self.biobert_pipeline:
            results['biobert'] = await self._extract_bert_entities(
                temp_doc, self.biobert_pipeline, "BioBERT"
            )
        
        if self.use_clinical_bert and self.clinical_pipeline:
            results['clinical_bert'] = await self._extract_bert_entities(
                temp_doc, self.clinical_pipeline, "ClinicalBERT"
            )
        
        return results
