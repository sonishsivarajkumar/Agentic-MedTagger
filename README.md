# Agentic MedTagger

A production-ready Python-native clinical NLP framework inspired by Mayo Clinic's MedTagger, implementing all major features for clinical text processing.

## Overview

Agentic MedTagger is a comprehensive clinical NLP framework that combines dictionary-based matching, pattern-based information extraction, machine learning-based NER, section detection, assertion/negation analysis, and OMOP/UMLS concept normalization. Built with modern Python practices, it provides high-performance processing of clinical documents with extensive configurability and extensibility.

The framework processes clinical text through a configurable pipeline that can extract entities, detect document sections, classify assertions, and normalize concepts to standard vocabularies. All components work together seamlessly while maintaining modular architecture for easy customization.

## Features

### Core Clinical NLP Components

- **Dictionary-Based Matching**: High-performance flashtext matching against CSV, JSON, MedLex, UMLS, and OMOP dictionaries
- **Pattern-Based Information Extraction**: Regex and spaCy patterns for clinical entities (vital signs, medications, demographics)
- **ML-Based Named Entity Recognition**: spaCy and Transformers integration with BioBERT/ClinicalBERT support
- **Section Detection**: Automatic detection of clinical document sections (Chief Complaint, HPI, Assessment, etc.)
- **Assertion & Negation**: pyConText-style analysis for clinical context understanding
- **OMOP/UMLS Normalization**: Concept mapping to standard medical vocabularies

### Technical Features

- **Production-Ready Architecture**: Modern Python packaging with comprehensive testing (27 tests)
- **High Performance**: ~0.1s processing time per document with async support
- **Configurable Pipeline**: Enable/disable components as needed
- **Extensible Design**: Plugin architecture for custom annotators
- **Multiple Output Formats**: Structured JSON annotations with rich metadata
- **Docker Ready**: Containerized deployment with CI/CD pipeline
- **Comprehensive Documentation**: Examples, demos, and API documentation

## Architecture

### Core Components

- **MedTaggerPipeline**: Orchestrates all processing components with configurable workflows
- **DictionaryMatcher**: Fast clinical terminology matching using flashtext
- **MedTaggerIE**: Pattern-based information extraction for clinical entities
- **MedTaggerML**: Machine learning-based NER using spaCy and Transformers
- **SectionDetector**: Clinical document structure recognition
- **AssertionNegationDetector**: Clinical context analysis (optional pyConText integration)
- **OMOPUMLSNormalizer**: Concept normalization to standard vocabularies

### Processing Pipeline

1. **Document Ingestion**: Clinical text preprocessing and validation
2. **Section Detection**: Identify document structure and clinical sections
3. **Dictionary Matching**: Fast terminology lookup against clinical vocabularies
4. **Pattern Extraction**: Regex and rule-based entity extraction
5. **ML-Based NER**: Deep learning entity recognition with clinical models
6. **Assertion Analysis**: Context classification (negation, uncertainty, experiencer)
7. **Concept Normalization**: Map entities to OMOP/UMLS standard concepts
8. **Structured Output**: Generate comprehensive JSON annotations with metadata

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/sonishsivarajkumar/Agentic-MedTagger.git
cd Agentic-MedTagger

# Install dependencies
pip install -r requirements.txt

# Download spaCy models
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_trf
```

### Basic Usage

```python
from agentic_medtagger.core.pipeline import create_medtagger_pipeline
from agentic_medtagger.core.document import Document

# Create pipeline with default configuration
pipeline = create_medtagger_pipeline()

# Process clinical text
text = """
CHIEF COMPLAINT: Chest pain and shortness of breath

HISTORY OF PRESENT ILLNESS:
Patient is a 65-year-old male with diabetes who presents with chest pain.
Blood pressure is 140/90 mmHg. Heart rate is 88 bpm.
"""

document = Document(text=text, document_id="clinical_note_001")
processed_doc = await pipeline.process_document(document)

# Access annotations
for annotation in processed_doc.annotations:
    print(f"{annotation.text} -> {annotation.label} (confidence: {annotation.confidence})")
```

### Run the Demo

```bash
# Run comprehensive demo showing all features
python demo_medtagger.py
```

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/agentic_medtagger --cov-report=html
```

## Configuration

The pipeline is highly configurable. See `config/medtagger_config.py` for examples:

```python
config = {
    'enable_section_detection': True,
    'enable_dictionary_matching': True,
    'enable_pattern_extraction': True,
    'enable_ml_ner': True,
    'enable_assertion_negation': False,  # Optional pyConText
    'enable_normalization': True,
    'dictionary_matcher': {
        'dictionaries': [
            {
                'name': 'clinical_terms',
                'type': 'csv',
                'path': 'data/clinical_terms.csv',
                'key_column': 'term'
            }
        ]
    },
    'omop_umls_normalizer': {
        'local_mappings': {
            'diabetes': {
                'concept_id': '201826',
                'concept_name': 'Type 2 diabetes mellitus',
                'vocabulary': 'SNOMED'
            }
        }
    }
}
```

## Performance

- **Processing Speed**: ~0.1s per clinical document
- **Memory Efficient**: Optimized for clinical workflows
- **Scalable**: Async processing support for high throughput
- **Test Coverage**: 27 comprehensive tests ensuring reliability
- **Production Ready**: Docker containerization and CI/CD pipeline

## Example Output

```json
{
  "document_id": "clinical_note_001",
  "annotations": [
    {
      "id": "ann_001",
      "span": {"start": 45, "end": 55, "text": "chest pain"},
      "label": "SYMPTOM",
      "annotation_type": "entity",
      "confidence": 0.95,
      "annotator": "medtagger_ie",
      "attributes": {
        "section": "CHIEF_COMPLAINT",
        "pattern_matched": "chest_pain_pattern"
      }
    },
    {
      "id": "ann_002", 
      "span": {"start": 120, "end": 132, "text": "140/90 mmHg"},
      "label": "BLOOD_PRESSURE",
      "annotation_type": "vital_sign",
      "confidence": 1.0,
      "annotator": "medtagger_ie",
      "attributes": {
        "systolic": "140",
        "diastolic": "90",
        "unit": "mmHg"
      }
    }
  ]
}
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use Agentic MedTagger in your research, please cite:

```bibtex
@software{agentic_medtagger2025,
  title={Agentic MedTagger: A Python-Native Clinical NLP Framework},
  author={Sonish Sivarajkumar},
  year={2025},
  url={https://github.com/sonishsivarajkumar/Agentic-MedTagger}
}
```
