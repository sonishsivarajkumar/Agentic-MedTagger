"""
Assertion and negation detection component using pyConText.
Identifies negated, uncertain, and historical medical concepts.
"""

import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
import spacy

try:
    from pycontext import ConTextComponent
    PYCONTEXT_AVAILABLE = True
except ImportError:
    PYCONTEXT_AVAILABLE = False
    ConTextComponent = None

from ..core.pipeline import PipelineComponent
from ..core.document import Document
from ..core.annotation import Annotation, Span


logger = logging.getLogger(__name__)


class AssertionNegationDetector(PipelineComponent):
    """
    Clinical assertion and negation detection using pyConText.
    Determines if medical concepts are negated, uncertain, or historical.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("assertion_negation", config)
        
        # spaCy model for text processing
        self.nlp_model = self.config.get('nlp_model', 'en_core_web_sm')
        self.nlp = None
        
        # pyConText configuration
        self.rules_path = self.config.get('rules_path', None)
        self.use_default_rules = self.config.get('use_default_rules', True)
        
        # Context categories to detect
        self.context_categories = self.config.get('context_categories', [
            'NEGATED_EXISTENCE',
            'UNCERTAIN',
            'HISTORICAL',
            'HYPOTHETICAL',
            'EXPERIENCED_BY_PATIENT',
            'EXPERIENCED_BY_FAMILY'
        ])
        
        # Target annotation types to process
        self.target_labels = self.config.get('target_labels', [
            'DISEASE', 'SYMPTOM', 'MEDICATION', 'PROCEDURE', 'TEST', 'CONDITION'
        ])
        
        # Confidence thresholds
        self.min_confidence = self.config.get('min_confidence', 0.7)
        
        # Custom rules
        self.custom_rules = self.config.get('custom_rules', {})
    
    async def initialize(self):
        """Initialize assertion/negation detector."""
        logger.info(f"Initializing {self.name}")
        
        if not PYCONTEXT_AVAILABLE:
            logger.warning("pyConText not available. Assertion/negation detection will be disabled.")
            return
        
        # Load spaCy model
        try:
            self.nlp = spacy.load(self.nlp_model)
        except OSError:
            logger.warning(f"Could not load spaCy model {self.nlp_model}, using blank model")
            self.nlp = spacy.blank("en")
        
        # Add pyConText component
        try:
            # Add ConText component to spaCy pipeline
            context_component = ConTextComponent(
                self.nlp,
                rules_path=self.rules_path,
                use_default_rules=self.use_default_rules
            )
            self.nlp.add_pipe(context_component)
            
            logger.info("Added pyConText component to spaCy pipeline")
        except Exception as e:
            logger.error(f"Failed to initialize pyConText: {e}")
            raise
        
        # Load custom rules if provided
        if self.custom_rules:
            self._load_custom_rules()
        
        logger.info(f"Initialized {self.name} with pyConText")
    
    def _load_custom_rules(self):
        """Load custom ConText rules."""
        try:
            # Add custom rules to the ConText component
            context_component = self.nlp.get_pipe("context")
            
            for category, rules in self.custom_rules.items():
                for rule in rules:
                    context_component.add_rule(rule, category)
            
            logger.info(f"Loaded {len(self.custom_rules)} custom rule categories")
        except Exception as e:
            logger.error(f"Failed to load custom rules: {e}")
    
    async def process(self, document: Document) -> Document:
        """Process document for assertion and negation."""
        logger.debug(f"Processing document with {self.name}")
        
        if not PYCONTEXT_AVAILABLE or not self.nlp:
            logger.warning("pyConText not available, skipping assertion/negation detection")
            return document
        
        # Process text with spaCy + pyConText
        doc = self.nlp(document.text)
        
        # Get existing annotations that should be processed
        target_annotations = self._get_target_annotations(document)
        
        # Process each target annotation
        for annotation in target_annotations:
            # Find the spaCy span that corresponds to this annotation
            spacy_span = self._find_spacy_span(doc, annotation)
            
            if spacy_span:
                # Check context for this span
                context_info = self._get_context_info(spacy_span)
                
                if context_info:
                    # Update annotation with context information
                    self._update_annotation_with_context(annotation, context_info)
        
        # Also process any entities found by spaCy/pyConText directly
        await self._process_spacy_entities(doc, document)
        
        return document
    
    def _get_target_annotations(self, document: Document) -> List[Annotation]:
        """Get annotations that should be processed for assertion/negation."""
        target_annotations = []
        
        for annotation in document.annotations:
            if annotation.label in self.target_labels:
                target_annotations.append(annotation)
        
        return target_annotations
    
    def _find_spacy_span(self, doc, annotation: Annotation):
        """Find the spaCy span that corresponds to an annotation."""
        # Find the spaCy span that best matches the annotation
        char_start = annotation.span.start
        char_end = annotation.span.end
        
        # Look for spaCy span that matches or overlaps
        for sent in doc.sents:
            for token in sent:
                if token.idx <= char_start < token.idx + len(token.text):
                    # Found starting token, now find ending token
                    end_token = token
                    while end_token.idx + len(end_token.text) < char_end and end_token.i < len(doc) - 1:
                        end_token = doc[end_token.i + 1]
                    
                    return doc[token.i:end_token.i + 1]
        
        return None
    
    def _get_context_info(self, span) -> Optional[Dict[str, Any]]:
        """Get context information for a spaCy span."""
        context_info = {}
        
        # Check if span has context attributes
        if hasattr(span, '_'):
            context_attrs = span._
            
            # Check for negation
            if hasattr(context_attrs, 'is_negated') and context_attrs.is_negated:
                context_info['negated'] = True
                context_info['negation_cues'] = getattr(context_attrs, 'negation_cues', [])
            
            # Check for uncertainty
            if hasattr(context_attrs, 'is_uncertain') and context_attrs.is_uncertain:
                context_info['uncertain'] = True
                context_info['uncertainty_cues'] = getattr(context_attrs, 'uncertainty_cues', [])
            
            # Check for historical context
            if hasattr(context_attrs, 'is_historical') and context_attrs.is_historical:
                context_info['historical'] = True
                context_info['historical_cues'] = getattr(context_attrs, 'historical_cues', [])
            
            # Check for hypothetical
            if hasattr(context_attrs, 'is_hypothetical') and context_attrs.is_hypothetical:
                context_info['hypothetical'] = True
                context_info['hypothetical_cues'] = getattr(context_attrs, 'hypothetical_cues', [])
            
            # Check for experiencer
            if hasattr(context_attrs, 'is_family') and context_attrs.is_family:
                context_info['experiencer'] = 'family'
                context_info['experiencer_cues'] = getattr(context_attrs, 'family_cues', [])
            else:
                context_info['experiencer'] = 'patient'
        
        return context_info if context_info else None
    
    def _update_annotation_with_context(self, annotation: Annotation, context_info: Dict[str, Any]):
        """Update annotation with context information."""
        # Update attributes with context information
        if 'context' not in annotation.attributes:
            annotation.attributes['context'] = {}
        
        annotation.attributes['context'].update(context_info)
        
        # Update label if negated
        if context_info.get('negated', False):
            annotation.label = f"NEGATED_{annotation.label}"
            annotation.confidence *= 0.9  # Slightly reduce confidence for negated items
        
        # Update label if uncertain
        if context_info.get('uncertain', False):
            annotation.label = f"UNCERTAIN_{annotation.label}"
            annotation.confidence *= 0.8  # Reduce confidence for uncertain items
        
        # Update label if historical
        if context_info.get('historical', False):
            annotation.label = f"HISTORICAL_{annotation.label}"
        
        # Update label if hypothetical
        if context_info.get('hypothetical', False):
            annotation.label = f"HYPOTHETICAL_{annotation.label}"
        
        # Update label if family history
        if context_info.get('experiencer') == 'family':
            annotation.label = f"FAMILY_{annotation.label}"
    
    async def _process_spacy_entities(self, doc, document: Document):
        """Process entities directly found by spaCy/pyConText."""
        for ent in doc.ents:
            # Check if this entity has context information
            context_info = self._get_context_info(ent)
            
            if context_info:
                # Create annotation with context
                annotation = Annotation(
                    id=f"context_{ent.start_char}_{ent.end_char}",
                    span=Span(ent.start_char, ent.end_char, ent.text),
                    label=ent.label_,
                    annotation_type='context_entity',
                    confidence=0.8,
                    annotator=self.name,
                    attributes={
                        'context': context_info,
                        'entity_type': ent.label_
                    }
                )
                
                # Apply context to label
                self._update_annotation_with_context(annotation, context_info)
                
                document.add_annotation(annotation)
    
    def analyze_context(self, text: str) -> Dict[str, Any]:
        """Analyze context in a text snippet."""
        doc = self.nlp(text)
        
        analysis = {
            'entities': [],
            'context_cues': [],
            'overall_context': {}
        }
        
        # Analyze entities
        for ent in doc.ents:
            entity_info = {
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'context': self._get_context_info(ent)
            }
            analysis['entities'].append(entity_info)
        
        # Extract context cues
        for token in doc:
            if hasattr(token, '_') and hasattr(token._, 'context_cues'):
                cue_info = {
                    'text': token.text,
                    'category': token._.context_cues,
                    'start': token.idx,
                    'end': token.idx + len(token.text)
                }
                analysis['context_cues'].append(cue_info)
        
        return analysis
    
    def get_negated_concepts(self, document: Document) -> List[Annotation]:
        """Get all negated concepts from document."""
        negated_concepts = []
        
        for annotation in document.annotations:
            if (annotation.annotator == self.name and 
                annotation.attributes.get('context', {}).get('negated', False)):
                negated_concepts.append(annotation)
        
        return negated_concepts
    
    def get_uncertain_concepts(self, document: Document) -> List[Annotation]:
        """Get all uncertain concepts from document."""
        uncertain_concepts = []
        
        for annotation in document.annotations:
            if (annotation.annotator == self.name and 
                annotation.attributes.get('context', {}).get('uncertain', False)):
                uncertain_concepts.append(annotation)
        
        return uncertain_concepts
    
    def get_historical_concepts(self, document: Document) -> List[Annotation]:
        """Get all historical concepts from document."""
        historical_concepts = []
        
        for annotation in document.annotations:
            if (annotation.annotator == self.name and 
                annotation.attributes.get('context', {}).get('historical', False)):
                historical_concepts.append(annotation)
        
        return historical_concepts
    
    def get_family_history(self, document: Document) -> List[Annotation]:
        """Get all family history concepts from document."""
        family_concepts = []
        
        for annotation in document.annotations:
            if (annotation.annotator == self.name and 
                annotation.attributes.get('context', {}).get('experiencer') == 'family'):
                family_concepts.append(annotation)
        
        return family_concepts
    
    def get_context_stats(self) -> Dict[str, Any]:
        """Get context detection statistics."""
        return {
            'nlp_model': self.nlp_model,
            'context_categories': self.context_categories,
            'target_labels': self.target_labels,
            'custom_rules': len(self.custom_rules)
        }
