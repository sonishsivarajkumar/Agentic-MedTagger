"""
OMOP/UMLS normalization component for mapping clinical concepts
to standardized vocabularies.
"""

import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
import json
import requests
from urllib.parse import quote
import pandas as pd

from ..core.pipeline import PipelineComponent
from ..core.document import Document
from ..core.annotation import Annotation, Span


logger = logging.getLogger(__name__)


class OMOPUMLSNormalizer(PipelineComponent):
    """
    OMOP/UMLS concept normalization component.
    Maps clinical entities to standardized concept codes.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("omop_umls_normalizer", config)
        
        # UMLS API configuration
        self.umls_api_key = self.config.get('umls_api_key', None)
        self.umls_base_url = self.config.get('umls_base_url', 'https://uts-ws.nlm.nih.gov/rest')
        self.umls_version = self.config.get('umls_version', 'current')
        
        # OMOP configuration
        self.omop_vocab_path = self.config.get('omop_vocab_path', None)
        self.omop_concept_table = None
        self.omop_concept_synonym_table = None
        
        # Normalization settings
        self.target_vocabularies = self.config.get('target_vocabularies', [
            'SNOMED', 'ICD10CM', 'ICD9CM', 'LOINC', 'RxNorm', 'CPT4'
        ])
        
        # Entity types to normalize
        self.target_labels = self.config.get('target_labels', [
            'DISEASE', 'SYMPTOM', 'MEDICATION', 'PROCEDURE', 'TEST', 'CONDITION'
        ])
        
        # Search configuration
        self.max_search_results = self.config.get('max_search_results', 10)
        self.min_similarity_score = self.config.get('min_similarity_score', 0.7)
        
        # Caching
        self.concept_cache = {}
        self.enable_cache = self.config.get('enable_cache', True)
        
        # Local concept mappings
        self.local_mappings = self.config.get('local_mappings', {})
    
    async def initialize(self):
        """Initialize OMOP/UMLS normalizer."""
        logger.info(f"Initializing {self.name}")
        
        # Load OMOP vocabulary if provided
        if self.omop_vocab_path:
            await self._load_omop_vocabulary()
        
        # Load local mappings
        if self.local_mappings:
            await self._load_local_mappings()
        
        # Test UMLS connection if API key provided
        if self.umls_api_key:
            await self._test_umls_connection()
        
        logger.info(f"Initialized {self.name}")
    
    async def _load_omop_vocabulary(self):
        """Load OMOP vocabulary tables."""
        try:
            vocab_path = Path(self.omop_vocab_path)
            
            # Load concept table
            concept_file = vocab_path / 'CONCEPT.csv'
            if concept_file.exists():
                self.omop_concept_table = pd.read_csv(concept_file, sep='\t', low_memory=False)
                logger.info(f"Loaded OMOP concept table with {len(self.omop_concept_table)} concepts")
            
            # Load concept synonym table
            synonym_file = vocab_path / 'CONCEPT_SYNONYM.csv'
            if synonym_file.exists():
                self.omop_concept_synonym_table = pd.read_csv(synonym_file, sep='\t', low_memory=False)
                logger.info(f"Loaded OMOP synonym table with {len(self.omop_concept_synonym_table)} synonyms")
        
        except Exception as e:
            logger.error(f"Failed to load OMOP vocabulary: {e}")
    
    async def _load_local_mappings(self):
        """Load local concept mappings."""
        try:
            if isinstance(self.local_mappings, str):
                # Load from file
                with open(self.local_mappings, 'r') as f:
                    self.local_mappings = json.load(f)
            
            logger.info(f"Loaded {len(self.local_mappings)} local concept mappings")
        
        except Exception as e:
            logger.error(f"Failed to load local mappings: {e}")
            self.local_mappings = {}
    
    async def _test_umls_connection(self):
        """Test UMLS API connection."""
        try:
            # Test search endpoint
            url = f"{self.umls_base_url}/search/{self.umls_version}"
            params = {
                'string': 'diabetes',
                'apiKey': self.umls_api_key,
                'returnIdType': 'concept'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                logger.info("UMLS API connection successful")
            else:
                logger.warning(f"UMLS API test failed: {response.status_code}")
        
        except Exception as e:
            logger.error(f"Failed to test UMLS connection: {e}")
    
    async def process(self, document: Document) -> Document:
        """Process document for concept normalization."""
        logger.debug(f"Processing document with {self.name}")
        
        # Get target annotations
        target_annotations = self._get_target_annotations(document)
        
        # Normalize each annotation
        for annotation in target_annotations:
            concept_mappings = await self._normalize_concept(annotation.text, annotation.label)
            
            if concept_mappings:
                # Update annotation with concept mappings
                if 'normalization' not in annotation.attributes:
                    annotation.attributes['normalization'] = {}
                
                annotation.attributes['normalization'].update({
                    'concepts': concept_mappings,
                    'normalized_by': self.name
                })
        
        return document
    
    def _get_target_annotations(self, document: Document) -> List[Annotation]:
        """Get annotations that should be normalized."""
        target_annotations = []
        
        for annotation in document.annotations:
            if annotation.label in self.target_labels:
                target_annotations.append(annotation)
        
        return target_annotations
    
    async def _normalize_concept(self, text: str, entity_type: str) -> List[Dict[str, Any]]:
        """Normalize a concept to standard vocabularies."""
        # Check cache first
        cache_key = f"{text}_{entity_type}"
        if self.enable_cache and cache_key in self.concept_cache:
            return self.concept_cache[cache_key]
        
        concept_mappings = []
        
        # Check local mappings first
        local_mapping = self._check_local_mappings(text, entity_type)
        if local_mapping:
            concept_mappings.extend(local_mapping)
        
        # Search OMOP vocabulary
        if self.omop_concept_table is not None:
            omop_mappings = await self._search_omop_concepts(text, entity_type)
            concept_mappings.extend(omop_mappings)
        
        # Search UMLS
        if self.umls_api_key:
            umls_mappings = await self._search_umls_concepts(text, entity_type)
            concept_mappings.extend(umls_mappings)
        
        # Deduplicate and sort by score
        concept_mappings = self._deduplicate_concepts(concept_mappings)
        concept_mappings.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # Cache result
        if self.enable_cache:
            self.concept_cache[cache_key] = concept_mappings
        
        return concept_mappings[:self.max_search_results]
    
    def _check_local_mappings(self, text: str, entity_type: str) -> List[Dict[str, Any]]:
        """Check local concept mappings."""
        mappings = []
        
        # Exact match
        if text.lower() in self.local_mappings:
            mapping = self.local_mappings[text.lower()]
            mappings.append({
                'concept_id': mapping.get('concept_id'),
                'concept_name': mapping.get('concept_name', text),
                'vocabulary': mapping.get('vocabulary', 'Local'),
                'score': 1.0,
                'source': 'local_mapping'
            })
        
        return mappings
    
    async def _search_omop_concepts(self, text: str, entity_type: str) -> List[Dict[str, Any]]:
        """Search OMOP concept table."""
        mappings = []
        
        try:
            if self.omop_concept_table is None:
                return mappings
            
            # Search in concept names
            concept_matches = self.omop_concept_table[
                self.omop_concept_table['concept_name'].str.contains(
                    text, case=False, na=False
                )
            ]
            
            # Search in synonyms if available
            if self.omop_concept_synonym_table is not None:
                synonym_matches = self.omop_concept_synonym_table[
                    self.omop_concept_synonym_table['concept_synonym_name'].str.contains(
                        text, case=False, na=False
                    )
                ]
                
                # Join with concept table
                if not synonym_matches.empty:
                    synonym_concepts = self.omop_concept_table[
                        self.omop_concept_table['concept_id'].isin(
                            synonym_matches['concept_id']
                        )
                    ]
                    concept_matches = pd.concat([concept_matches, synonym_concepts]).drop_duplicates()
            
            # Convert to mappings
            for _, row in concept_matches.head(self.max_search_results).iterrows():
                mapping = {
                    'concept_id': str(row['concept_id']),
                    'concept_name': row['concept_name'],
                    'vocabulary': row.get('vocabulary_id', 'OMOP'),
                    'domain': row.get('domain_id', 'Unknown'),
                    'concept_class': row.get('concept_class_id', 'Unknown'),
                    'score': self._calculate_similarity_score(text, row['concept_name']),
                    'source': 'omop'
                }
                
                if mapping['score'] >= self.min_similarity_score:
                    mappings.append(mapping)
        
        except Exception as e:
            logger.error(f"Error searching OMOP concepts: {e}")
        
        return mappings
    
    async def _search_umls_concepts(self, text: str, entity_type: str) -> List[Dict[str, Any]]:
        """Search UMLS concepts via API."""
        mappings = []
        
        try:
            if not self.umls_api_key:
                return mappings
            
            # Search for concepts
            url = f"{self.umls_base_url}/search/{self.umls_version}"
            params = {
                'string': text,
                'apiKey': self.umls_api_key,
                'returnIdType': 'concept',
                'pageSize': self.max_search_results
            }
            
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                for result in data.get('result', {}).get('results', []):
                    concept_ui = result.get('ui')
                    concept_name = result.get('name', '')
                    
                    # Get concept details
                    concept_details = await self._get_umls_concept_details(concept_ui)
                    
                    mapping = {
                        'concept_id': concept_ui,
                        'concept_name': concept_name,
                        'vocabulary': 'UMLS',
                        'score': self._calculate_similarity_score(text, concept_name),
                        'source': 'umls',
                        'details': concept_details
                    }
                    
                    if mapping['score'] >= self.min_similarity_score:
                        mappings.append(mapping)
        
        except Exception as e:
            logger.error(f"Error searching UMLS concepts: {e}")
        
        return mappings
    
    async def _get_umls_concept_details(self, concept_id: str) -> Dict[str, Any]:
        """Get detailed information about a UMLS concept."""
        try:
            url = f"{self.umls_base_url}/content/{self.umls_version}/CUI/{concept_id}"
            params = {'apiKey': self.umls_api_key}
            
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                return response.json().get('result', {})
        
        except Exception as e:
            logger.error(f"Error getting UMLS concept details: {e}")
        
        return {}
    
    def _calculate_similarity_score(self, text1: str, text2: str) -> float:
        """Calculate similarity score between two text strings."""
        # Simple similarity based on common words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _deduplicate_concepts(self, concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate concepts."""
        seen_concepts = set()
        unique_concepts = []
        
        for concept in concepts:
            concept_key = (concept.get('concept_id'), concept.get('vocabulary'))
            
            if concept_key not in seen_concepts:
                seen_concepts.add(concept_key)
                unique_concepts.append(concept)
        
        return unique_concepts
    
    def get_concept_by_id(self, concept_id: str, vocabulary: str = None) -> Optional[Dict[str, Any]]:
        """Get concept information by ID."""
        if vocabulary == 'OMOP' and self.omop_concept_table is not None:
            matches = self.omop_concept_table[
                self.omop_concept_table['concept_id'] == int(concept_id)
            ]
            
            if not matches.empty:
                row = matches.iloc[0]
                return {
                    'concept_id': str(row['concept_id']),
                    'concept_name': row['concept_name'],
                    'vocabulary': row.get('vocabulary_id', 'OMOP'),
                    'domain': row.get('domain_id', 'Unknown'),
                    'concept_class': row.get('concept_class_id', 'Unknown')
                }
        
        return None
    
    def get_normalized_concepts(self, document: Document) -> Dict[str, List[Dict[str, Any]]]:
        """Get all normalized concepts from document."""
        normalized_concepts = {}
        
        for annotation in document.annotations:
            if 'normalization' in annotation.attributes:
                concepts = annotation.attributes['normalization'].get('concepts', [])
                
                if concepts:
                    if annotation.label not in normalized_concepts:
                        normalized_concepts[annotation.label] = []
                    
                    normalized_concepts[annotation.label].extend(concepts)
        
        return normalized_concepts
    
    def add_local_mapping(self, text: str, concept_id: str, concept_name: str, vocabulary: str):
        """Add a local concept mapping."""
        self.local_mappings[text.lower()] = {
            'concept_id': concept_id,
            'concept_name': concept_name,
            'vocabulary': vocabulary
        }
    
    def get_normalization_stats(self) -> Dict[str, Any]:
        """Get normalization statistics."""
        return {
            'target_vocabularies': self.target_vocabularies,
            'target_labels': self.target_labels,
            'local_mappings': len(self.local_mappings),
            'cache_size': len(self.concept_cache),
            'omop_loaded': self.omop_concept_table is not None,
            'umls_enabled': self.umls_api_key is not None
        }
