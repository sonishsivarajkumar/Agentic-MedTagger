"""
Configuration for MedTagger pipeline components.
"""

# MedTagger Pipeline Configuration
medtagger_config = {
    # Pipeline-wide settings
    'enable_section_detection': True,
    'enable_dictionary_matching': True,
    'enable_pattern_extraction': True,
    'enable_ml_ner': True,
    'enable_assertion_negation': True,
    'enable_normalization': True,
    
    # Section Detection Configuration
    'section_detector': {
        'case_sensitive': False,
        'custom_patterns': {
            'ALLERGIES': [
                r'\b(?:allergies|allergy|nkda)\s*:',
                r'\bnkda\s*:',
                r'\bno\s+known\s+drug\s+allergies\s*:'
            ]
        }
    },
    
    # Dictionary Matching Configuration
    'dictionary_matcher': {
        'case_sensitive': False,
        'overlapping_matches': False,
        'longest_match_only': True,
        'dictionaries': [
            {
                'name': 'medical_conditions',
                'type': 'csv',
                'path': 'data/dictionaries/conditions.csv',
                'key_column': 'term',
                'value_column': 'category',
                'enabled': True
            },
            {
                'name': 'medications',
                'type': 'csv', 
                'path': 'data/dictionaries/medications.csv',
                'key_column': 'drug_name',
                'value_column': 'drug_class',
                'enabled': True
            }
        ]
    },
    
    # Pattern-based Information Extraction Configuration  
    'medtagger_ie': {
        'use_clinical_patterns': True,
        'case_sensitive': False,
        'overlapping_matches': False,
        'max_matches_per_pattern': 100,
        'nlp_model': 'en_core_web_sm'
    },
    
    # ML-based NER Configuration
    'medtagger_ml': {
        'use_spacy': True,
        'use_biobert': False,  # Set to True if BioBERT models are available
        'use_clinical_bert': False,  # Set to True if Clinical BERT models are available
        'spacy_model': 'en_core_web_sm',
        'biobert_model': 'dmis-lab/biobert-base-cased-v1.1',
        'clinical_bert_model': 'emilyalsentzer/Bio_ClinicalBERT',
        'min_confidence': 0.5,
        'aggregate_overlapping': True,
        'prefer_longer_entities': True
    },
    
    # Assertion and Negation Detection Configuration
    'assertion_negation': {
        'nlp_model': 'en_core_web_sm',
        'use_default_rules': True,
        'rules_path': None,  # Path to custom pyConText rules
        'target_labels': [
            'DISEASE', 'SYMPTOM', 'MEDICATION', 'PROCEDURE', 'TEST', 'CONDITION'
        ],
        'min_confidence': 0.7
    },
    
    # OMOP/UMLS Normalization Configuration
    'omop_umls_normalizer': {
        'umls_api_key': None,  # Set your UMLS API key here
        'umls_base_url': 'https://uts-ws.nlm.nih.gov/rest',
        'umls_version': 'current',
        'omop_vocab_path': None,  # Path to OMOP vocabulary files
        'target_vocabularies': [
            'SNOMED', 'ICD10CM', 'ICD9CM', 'LOINC', 'RxNorm', 'CPT4'
        ],
        'target_labels': [
            'DISEASE', 'SYMPTOM', 'MEDICATION', 'PROCEDURE', 'TEST', 'CONDITION'
        ],
        'max_search_results': 10,
        'min_similarity_score': 0.7,
        'enable_cache': True,
        'local_mappings': {
            # Common medical concept mappings
            'diabetes': {
                'concept_id': '201826',
                'concept_name': 'Type 2 diabetes mellitus',
                'vocabulary': 'SNOMED'
            },
            'hypertension': {
                'concept_id': '38341003',
                'concept_name': 'Hypertensive disorder',
                'vocabulary': 'SNOMED'
            },
            'myocardial infarction': {
                'concept_id': '22298006',
                'concept_name': 'Myocardial infarction',
                'vocabulary': 'SNOMED'
            },
            'pneumonia': {
                'concept_id': '233604007',
                'concept_name': 'Pneumonia',
                'vocabulary': 'SNOMED'
            },
            'aspirin': {
                'concept_id': '1191',
                'concept_name': 'Aspirin',
                'vocabulary': 'RxNorm'
            },
            'metformin': {
                'concept_id': '6809',
                'concept_name': 'Metformin',
                'vocabulary': 'RxNorm'
            }
        }
    }
}


# Default configuration for quick setup
default_config = {
    'enable_section_detection': True,
    'enable_dictionary_matching': True,
    'enable_pattern_extraction': True,
    'enable_ml_ner': True,
    'enable_assertion_negation': False,  # Disabled by default due to pycontext dependency
    'enable_normalization': True,
    
    'medtagger_ie': {
        'use_clinical_patterns': True,
        'nlp_model': 'en_core_web_sm'
    },
    
    'medtagger_ml': {
        'use_spacy': True,
        'use_biobert': False,
        'use_clinical_bert': False,
        'spacy_model': 'en_core_web_sm'
    },
    
    'omop_umls_normalizer': {
        'local_mappings': {
            'diabetes': {
                'concept_id': '201826',
                'concept_name': 'Type 2 diabetes mellitus',
                'vocabulary': 'SNOMED'
            },
            'hypertension': {
                'concept_id': '38341003',
                'concept_name': 'Hypertensive disorder',
                'vocabulary': 'SNOMED'
            }
        }
    }
}


# Minimal configuration for testing
minimal_config = {
    'enable_section_detection': True,
    'enable_dictionary_matching': False,
    'enable_pattern_extraction': True,
    'enable_ml_ner': False,
    'enable_assertion_negation': False,
    'enable_normalization': False
}
