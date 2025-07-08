"""
Dictionary-based annotation component inspired by Mayo Clinic MedTagger.
"""

import logging
from typing import Dict, List, Any, Optional, Set
import csv
import pandas as pd
from pathlib import Path
import json

from flashtext import KeywordProcessor
from ..core.pipeline import PipelineComponent
from ..core.document import Document
from ..core.annotation import Annotation, Span


logger = logging.getLogger(__name__)


class DictionaryMatcher(PipelineComponent):
    """
    Dictionary-based matcher using flashtext for fast string matching.
    Inspired by Mayo Clinic MedTagger's dictionary indexing component.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("dictionary_matcher", config)
        
        # Initialize keyword processor
        self.keyword_processor = KeywordProcessor(case_sensitive=False)
        
        # Dictionary sources
        self.dictionaries = self.config.get('dictionaries', [])
        
        # Normalization options
        self.case_sensitive = self.config.get('case_sensitive', False)
        self.overlapping_matches = self.config.get('overlapping_matches', False)
        self.longest_match_only = self.config.get('longest_match_only', True)
        
        # OMOP/UMLS integration
        self.omop_enabled = self.config.get('omop_enabled', False)
        self.umls_enabled = self.config.get('umls_enabled', False)
        
        # Performance tracking
        self.dictionary_stats = {}
    
    async def initialize(self):
        """Initialize dictionaries and keyword processor."""
        logger.info(f"Initializing {self.name} with {len(self.dictionaries)} dictionaries")
        
        if self.case_sensitive:
            self.keyword_processor = KeywordProcessor(case_sensitive=True)
        
        total_terms = 0
        for dict_config in self.dictionaries:
            terms_loaded = await self._load_dictionary(dict_config)
            total_terms += terms_loaded
            
        logger.info(f"Loaded {total_terms} total terms across all dictionaries")
        
        # Configure processor options  
        # Note: flashtext doesn't have set_longest_first method
        # Will handle longest match manually in post-processing if needed
    
    async def _load_dictionary(self, dict_config: Dict[str, Any]) -> int:
        """Load a single dictionary from various formats."""
        dict_type = dict_config.get('type', 'csv')
        file_path = dict_config.get('path')
        label = dict_config.get('label', 'ENTITY')
        
        if not file_path or not Path(file_path).exists():
            logger.warning(f"Dictionary file not found: {file_path}")
            return 0
        
        terms_loaded = 0
        
        try:
            if dict_type == 'csv':
                terms_loaded = await self._load_csv_dictionary(file_path, label, dict_config)
            elif dict_type == 'json':
                terms_loaded = await self._load_json_dictionary(file_path, label, dict_config)
            elif dict_type == 'medlex':
                terms_loaded = await self._load_medlex_dictionary(file_path, label, dict_config)
            elif dict_type == 'umls':
                terms_loaded = await self._load_umls_dictionary(file_path, label, dict_config)
            elif dict_type == 'omop':
                terms_loaded = await self._load_omop_dictionary(file_path, label, dict_config)
            else:
                logger.warning(f"Unsupported dictionary type: {dict_type}")
                
            self.dictionary_stats[label] = {
                'type': dict_type,
                'file_path': file_path,
                'terms_loaded': terms_loaded
            }
            
            logger.info(f"Loaded {terms_loaded} terms from {dict_type} dictionary: {label}")
            
        except Exception as e:
            logger.error(f"Failed to load dictionary {file_path}: {e}")
            
        return terms_loaded
    
    async def _load_csv_dictionary(self, file_path: str, label: str, config: Dict[str, Any]) -> int:
        """Load dictionary from CSV file."""
        term_column = config.get('term_column', 'term')
        concept_column = config.get('concept_column', 'concept')
        semantic_type_column = config.get('semantic_type_column', 'semantic_type')
        
        terms_loaded = 0
        
        try:
            df = pd.read_csv(file_path, dtype=str)
            
            for _, row in df.iterrows():
                term = row.get(term_column, '').strip()
                if not term:
                    continue
                
                # Create metadata for the term
                metadata = {
                    'label': label,
                    'concept': row.get(concept_column, ''),
                    'semantic_type': row.get(semantic_type_column, ''),
                    'source': 'csv'
                }
                
                # Add additional columns as metadata
                for col in df.columns:
                    if col not in [term_column, concept_column, semantic_type_column]:
                        metadata[col] = row.get(col, '')
                
                # Add to keyword processor
                self.keyword_processor.add_keyword(term, (label, metadata))
                terms_loaded += 1
                
        except Exception as e:
            logger.error(f"Error loading CSV dictionary {file_path}: {e}")
            
        return terms_loaded
    
    async def _load_json_dictionary(self, file_path: str, label: str, config: Dict[str, Any]) -> int:
        """Load dictionary from JSON file."""
        terms_loaded = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        term = item.get('term', '').strip()
                        if term:
                            metadata = {
                                'label': label,
                                'source': 'json',
                                **item
                            }
                            self.keyword_processor.add_keyword(term, (label, metadata))
                            terms_loaded += 1
            elif isinstance(data, dict):
                for term, metadata in data.items():
                    if term.strip():
                        full_metadata = {
                            'label': label,
                            'source': 'json',
                            **(metadata if isinstance(metadata, dict) else {})
                        }
                        self.keyword_processor.add_keyword(term, (label, full_metadata))
                        terms_loaded += 1
                        
        except Exception as e:
            logger.error(f"Error loading JSON dictionary {file_path}: {e}")
            
        return terms_loaded
    
    async def _load_medlex_dictionary(self, file_path: str, label: str, config: Dict[str, Any]) -> int:
        """Load MedLex-format dictionary."""
        # MedLex format: term|concept_id|semantic_type|...
        terms_loaded = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('|')
                    if len(parts) >= 2:
                        term = parts[0].strip()
                        concept_id = parts[1].strip()
                        semantic_type = parts[2].strip() if len(parts) > 2 else ''
                        
                        if term:
                            metadata = {
                                'label': label,
                                'concept_id': concept_id,
                                'semantic_type': semantic_type,
                                'source': 'medlex'
                            }
                            
                            self.keyword_processor.add_keyword(term, (label, metadata))
                            terms_loaded += 1
                            
        except Exception as e:
            logger.error(f"Error loading MedLex dictionary {file_path}: {e}")
            
        return terms_loaded
    
    async def _load_umls_dictionary(self, file_path: str, label: str, config: Dict[str, Any]) -> int:
        """Load UMLS MRCONSO format dictionary."""
        # UMLS MRCONSO format: CUI|LAT|TS|LUI|STT|SUI|ISPREF|AUI|SAUI|SCUI|SDUI|SAB|TTY|CODE|STR|SRL|SUPPRESS|CVF
        semantic_types = set(config.get('semantic_types', []))
        languages = set(config.get('languages', ['ENG']))
        
        terms_loaded = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('|')
                    if len(parts) >= 15:
                        cui = parts[0]
                        language = parts[1]
                        term = parts[14]
                        source = parts[11]
                        
                        # Filter by language
                        if languages and language not in languages:
                            continue
                        
                        # Filter by semantic types if specified
                        # Note: This is simplified - real UMLS would need MRSTY lookup
                        
                        if term.strip():
                            metadata = {
                                'label': label,
                                'cui': cui,
                                'language': language,
                                'source': source,
                                'umls_source': 'mrconso'
                            }
                            
                            self.keyword_processor.add_keyword(term, (label, metadata))
                            terms_loaded += 1
                            
        except Exception as e:
            logger.error(f"Error loading UMLS dictionary {file_path}: {e}")
            
        return terms_loaded
    
    async def _load_omop_dictionary(self, file_path: str, label: str, config: Dict[str, Any]) -> int:
        """Load OMOP concept dictionary."""
        terms_loaded = 0
        
        try:
            df = pd.read_csv(file_path, sep='\t', dtype=str)
            
            for _, row in df.iterrows():
                concept_name = row.get('concept_name', '').strip()
                if not concept_name:
                    continue
                
                metadata = {
                    'label': label,
                    'concept_id': row.get('concept_id', ''),
                    'domain_id': row.get('domain_id', ''),
                    'vocabulary_id': row.get('vocabulary_id', ''),
                    'concept_class_id': row.get('concept_class_id', ''),
                    'source': 'omop'
                }
                
                self.keyword_processor.add_keyword(concept_name, (label, metadata))
                terms_loaded += 1
                
                # Also add synonyms if available
                synonyms = row.get('concept_synonym_name', '')
                if synonyms:
                    for synonym in synonyms.split('|'):
                        if synonym.strip():
                            self.keyword_processor.add_keyword(synonym.strip(), (label, metadata))
                            terms_loaded += 1
                            
        except Exception as e:
            logger.error(f"Error loading OMOP dictionary {file_path}: {e}")
            
        return terms_loaded
    
    async def process(self, document: Document) -> Document:
        """Process document and add dictionary matches."""
        if not self.keyword_processor:
            await self.initialize()
        
        text = document.text
        
        # Extract keywords with spans
        keywords_found = self.keyword_processor.extract_keywords(text, span_info=True)
        
        annotations_added = 0
        
        for keyword, start_idx, end_idx in keywords_found:
            # Get the metadata for this keyword
            match_data = self.keyword_processor.get_keyword(keyword)
            if not match_data:
                continue
                
            label, metadata = match_data
            
            # Create span
            span = Span(
                start=start_idx,
                end=end_idx,
                text=text[start_idx:end_idx]
            )
            
            # Create annotation
            annotation = Annotation(
                id=f"dict_match_{start_idx}_{end_idx}_{annotations_added}",
                span=span,
                label=label,
                annotation_type="dictionary_match",
                annotator=self.name,
                confidence=1.0,  # Dictionary matches have full confidence
                attributes=metadata
            )
            
            # Add normalized concept information
            if 'concept_id' in metadata:
                annotation.normalized_concept = metadata['concept_id']
            if 'cui' in metadata:
                annotation.umls_cui = metadata['cui']
            if 'semantic_type' in metadata:
                annotation.umls_semantic_types = [metadata['semantic_type']]
            
            document.add_annotation(annotation)
            annotations_added += 1
        
        # Mark section as processed
        document.mark_section_processed("dictionary_matching")
        
        logger.debug(f"Added {annotations_added} dictionary matches to document {document.document_id}")
        
        return document
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dictionary matching statistics."""
        return {
            'dictionaries_loaded': len(self.dictionary_stats),
            'dictionary_details': self.dictionary_stats,
            'total_terms': sum(d['terms_loaded'] for d in self.dictionary_stats.values()),
            'configuration': {
                'case_sensitive': self.case_sensitive,
                'overlapping_matches': self.overlapping_matches,
                'longest_match_only': self.longest_match_only
            }
        }
    
    def add_custom_terms(self, terms: List[Dict[str, Any]], label: str = "CUSTOM"):
        """Add custom terms at runtime."""
        terms_added = 0
        
        for term_data in terms:
            term = term_data.get('term', '').strip()
            if not term:
                continue
                
            metadata = {
                'label': label,
                'source': 'custom',
                **term_data
            }
            
            self.keyword_processor.add_keyword(term, (label, metadata))
            terms_added += 1
        
        logger.info(f"Added {terms_added} custom terms with label {label}")
        return terms_added
