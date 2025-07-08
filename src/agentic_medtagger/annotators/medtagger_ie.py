"""
MedTaggerIE: Pattern-based Information Extraction component.
Inspired by Mayo Clinic MedTagger's pattern-based IE module.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Set, Pattern, Tuple
from pathlib import Path
import json
import spacy
from spacy.tokens import Doc, Span as SpacySpan

from ..core.pipeline import PipelineComponent
from ..core.document import Document
from ..core.annotation import Annotation, Span


logger = logging.getLogger(__name__)


class MedTaggerIE(PipelineComponent):
    """
    Pattern-based Information Extraction component using regular expressions
    and spaCy patterns for clinical text analysis.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("medtagger_ie", config)
        
        # Pattern definitions
        self.regex_patterns = self.config.get('regex_patterns', [])
        self.spacy_patterns = self.config.get('spacy_patterns', [])
        
        # Compiled regex patterns
        self.compiled_patterns = {}
        
        # spaCy model for advanced pattern matching
        self.nlp_model = self.config.get('nlp_model', 'en_core_web_sm')
        self.nlp = None
        
        # Pattern matching options
        self.case_sensitive = self.config.get('case_sensitive', False)
        self.overlapping_matches = self.config.get('overlapping_matches', False)
        self.max_matches_per_pattern = self.config.get('max_matches_per_pattern', 100)
        
        # Clinical domain patterns
        self.clinical_patterns = self._get_default_clinical_patterns()
        
    def _get_default_clinical_patterns(self) -> List[Dict[str, Any]]:
        """Get default clinical patterns for common medical entities."""
        return [
            # Blood pressure patterns
            {
                'name': 'blood_pressure',
                'type': 'regex',
                'pattern': r'\b(\d{2,3})/(\d{2,3})\s*mmHg\b',
                'labels': ['VITAL_SIGN', 'BLOOD_PRESSURE'],
                'groups': ['systolic', 'diastolic']
            },
            # Heart rate patterns
            {
                'name': 'heart_rate',
                'type': 'regex', 
                'pattern': r'\b(?:HR|heart rate|pulse)[\s:]*(?:of\s*)?(\d{2,3})\s*(?:bpm|beats per minute)?\b',
                'labels': ['VITAL_SIGN', 'HEART_RATE'],
                'groups': ['rate']
            },
            # Temperature patterns
            {
                'name': 'temperature',
                'type': 'regex',
                'pattern': r'\b(?:temp|temperature)[\s:]*(\d{2,3}(?:\.\d)?)\s*(?:°F|°C|F|C|degrees?)\b',
                'labels': ['VITAL_SIGN', 'TEMPERATURE'],
                'groups': ['value']
            },
            # Medication dosage patterns
            {
                'name': 'medication_dosage',
                'type': 'regex',
                'pattern': r'\b(\d+(?:\.\d+)?)\s*(mg|g|mcg|µg|ml|cc|units?)\b',
                'labels': ['MEDICATION', 'DOSAGE'],
                'groups': ['amount', 'unit']
            },
            # Age patterns
            {
                'name': 'age',
                'type': 'regex',
                'pattern': r'\b(\d{1,3})\s*(?:year|yr|years|yrs)?\s*old\b',
                'labels': ['DEMOGRAPHIC', 'AGE'],
                'groups': ['age']
            },
            # Date patterns
            {
                'name': 'date',
                'type': 'regex',
                'pattern': r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b',
                'labels': ['TEMPORAL', 'DATE'],
                'groups': ['month', 'day', 'year']
            },
            # Lab values
            {
                'name': 'lab_value',
                'type': 'regex',
                'pattern': r'\b(?:glucose|creatinine|BUN|hemoglobin|hematocrit|WBC|RBC)[\s:]*(\d+(?:\.\d+)?)\s*(?:mg/dL|g/dL|K/µL|M/µL)?\b',
                'labels': ['LAB_VALUE'],
                'groups': ['value']
            },
            # Procedure patterns
            {
                'name': 'procedure',
                'type': 'spacy',
                'pattern': [
                    {'LOWER': {'IN': ['underwent', 'performed', 'had']}},
                    {'POS': {'IN': ['DET', 'ADJ']}, 'OP': '*'},
                    {'POS': 'NOUN', 'OP': '+'}
                ],
                'labels': ['PROCEDURE']
            },
            # Diagnosis patterns
            {
                'name': 'diagnosis',
                'type': 'spacy',
                'pattern': [
                    {'LOWER': {'IN': ['diagnosed', 'diagnosis', 'dx']}},
                    {'LOWER': {'IN': ['with', 'of']}, 'OP': '?'},
                    {'POS': {'IN': ['ADJ', 'NOUN']}, 'OP': '+'}
                ],
                'labels': ['DIAGNOSIS']
            }
        ]
    
    async def initialize(self):
        """Initialize the IE component."""
        logger.info(f"Initializing {self.name}")
        
        # Load spaCy model
        try:
            self.nlp = spacy.load(self.nlp_model)
            logger.info(f"Loaded spaCy model: {self.nlp_model}")
        except OSError:
            logger.warning(f"Could not load spaCy model {self.nlp_model}, using blank model")
            self.nlp = spacy.blank("en")
        
        # Compile regex patterns
        self._compile_patterns()
        
        # Add clinical patterns if enabled
        if self.config.get('use_clinical_patterns', True):
            self._add_clinical_patterns()
        
        logger.info(f"Initialized {len(self.compiled_patterns)} regex patterns")
    
    def _compile_patterns(self):
        """Compile regex patterns for efficiency."""
        flags = 0 if self.case_sensitive else re.IGNORECASE
        
        # Compile user-defined patterns
        for pattern_config in self.regex_patterns:
            pattern_name = pattern_config['name']
            pattern_str = pattern_config['pattern']
            
            try:
                compiled_pattern = re.compile(pattern_str, flags)
                self.compiled_patterns[pattern_name] = {
                    'pattern': compiled_pattern,
                    'config': pattern_config
                }
            except re.error as e:
                logger.error(f"Invalid regex pattern '{pattern_name}': {e}")
    
    def _add_clinical_patterns(self):
        """Add default clinical patterns."""
        flags = 0 if self.case_sensitive else re.IGNORECASE
        
        for pattern_config in self.clinical_patterns:
            if pattern_config['type'] == 'regex':
                pattern_name = pattern_config['name']
                pattern_str = pattern_config['pattern']
                
                try:
                    compiled_pattern = re.compile(pattern_str, flags)
                    self.compiled_patterns[pattern_name] = {
                        'pattern': compiled_pattern,
                        'config': pattern_config
                    }
                except re.error as e:
                    logger.error(f"Invalid clinical pattern '{pattern_name}': {e}")
    
    async def process(self, document: Document) -> Document:
        """Process document with pattern-based IE."""
        logger.debug(f"Processing document with {self.name}")
        
        # Extract patterns using regex
        await self._extract_regex_patterns(document)
        
        # Extract patterns using spaCy
        if self.nlp:
            await self._extract_spacy_patterns(document)
        
        return document
    
    async def _extract_regex_patterns(self, document: Document):
        """Extract entities using regex patterns."""
        text = document.text
        
        for pattern_name, pattern_data in self.compiled_patterns.items():
            pattern = pattern_data['pattern']
            config = pattern_data['config']
            
            matches = pattern.finditer(text)
            match_count = 0
            
            for match in matches:
                if match_count >= self.max_matches_per_pattern:
                    break
                
                # Create annotation
                start, end = match.span()
                matched_text = text[start:end]
                span = Span(start, end, matched_text)
                
                # Extract groups if defined
                groups = {}
                if 'groups' in config and match.groups():
                    for i, group_name in enumerate(config['groups']):
                        if i < len(match.groups()) and match.group(i + 1):
                            groups[group_name] = match.group(i + 1)
                
                # Create annotation with labels
                labels = config.get('labels', [pattern_name.upper()])
                for label in labels:
                    annotation = Annotation(
                        id=f"ie_{pattern_name}_{start}_{end}_{match_count}",
                        span=span,
                        label=label,
                        annotation_type="pattern_match",
                        confidence=1.0,
                        annotator=self.name,
                        attributes={
                            'pattern_name': pattern_name,
                            'pattern_type': 'regex',
                            'groups': groups
                        }
                    )
                    document.add_annotation(annotation)
                
                match_count += 1
    
    async def _extract_spacy_patterns(self, document: Document):
        """Extract entities using spaCy patterns."""
        if not self.nlp:
            return
        
        # Process text with spaCy
        doc = self.nlp(document.text)
        
        # Apply spaCy patterns
        for pattern_config in self.spacy_patterns + self.clinical_patterns:
            if pattern_config['type'] != 'spacy':
                continue
            
            pattern_name = pattern_config['name']
            pattern = pattern_config['pattern']
            
            # Create matcher
            from spacy.matcher import Matcher
            matcher = Matcher(self.nlp.vocab)
            matcher.add(pattern_name, [pattern])
            
            # Find matches
            matches = matcher(doc)
            
            for match_id, start, end in matches:
                span = doc[start:end]
                
                # Create annotation
                char_start = span.start_char
                char_end = span.end_char
                span_text = span.text
                annotation_span = Span(char_start, char_end, span_text)
                
                labels = pattern_config.get('labels', [pattern_name.upper()])
                for label in labels:
                    annotation = Annotation(
                        id=f"spacy_{pattern_name}_{char_start}_{char_end}",
                        span=annotation_span,
                        label=label,
                        annotation_type="spacy_pattern",
                        confidence=1.0,
                        annotator=self.name,
                        attributes={
                            'pattern_name': pattern_name,
                            'pattern_type': 'spacy',
                            'spacy_label': self.nlp.vocab.strings[match_id]
                        }
                    )
                    document.add_annotation(annotation)
    
    def load_patterns_from_file(self, file_path: str):
        """Load patterns from a JSON file."""
        try:
            with open(file_path, 'r') as f:
                patterns = json.load(f)
            
            if 'regex_patterns' in patterns:
                self.regex_patterns.extend(patterns['regex_patterns'])
            
            if 'spacy_patterns' in patterns:
                self.spacy_patterns.extend(patterns['spacy_patterns'])
            
            logger.info(f"Loaded patterns from {file_path}")
        except Exception as e:
            logger.error(f"Failed to load patterns from {file_path}: {e}")
    
    def add_pattern(self, pattern_config: Dict[str, Any]):
        """Add a single pattern configuration."""
        if pattern_config['type'] == 'regex':
            self.regex_patterns.append(pattern_config)
        elif pattern_config['type'] == 'spacy':
            self.spacy_patterns.append(pattern_config)
    
    def get_pattern_stats(self) -> Dict[str, Any]:
        """Get statistics about pattern matching."""
        return {
            'total_regex_patterns': len(self.regex_patterns),
            'total_spacy_patterns': len(self.spacy_patterns),
            'clinical_patterns': len(self.clinical_patterns),
            'compiled_patterns': len(self.compiled_patterns)
        }
