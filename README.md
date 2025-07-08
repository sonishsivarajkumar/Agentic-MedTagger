# Agentic MedTagger

An Agentic AI based Python-native clinical NLP framework inspired by MedTagger's speed and simplicity.

## Overview

Agentic MedTagger leverages spaCy for tokenization and flashtext for lightning-fast dictionary matching against user-supplied CSV or UMLS-derived lexica. Section detection, assertion, and negation are driven by configurable JSON/YAML rule sets and pyConText-style engines.

The core pipeline emits structured annotations in JSON or XMI, ready for downstream analysis or integration. An agentic layer powered by LangChain agents routes ambiguous mentions to an LLM for context-aware UMLS normalization.

## Features

- 🚀 **Fast Dictionary Matching**: Lightning-fast flashtext-based matching against CSV or UMLS lexica
- 🔍 **Advanced NLP Pipeline**: spaCy tokenization with configurable rule-based processing
- 🤖 **Agentic Layer**: LangChain-powered LLM integration for ambiguous mention resolution
- 📊 **Active Learning**: Low-confidence cases surfaced for human review and feedback
- 🌐 **Modern Web UI**: FastAPI backend with React frontend (Tailwind + shadcn/ui)
- 💾 **Persistent Storage**: PostgreSQL backend for annotations and user feedback
- 🔌 **Plugin Architecture**: Extensible via Python entry points
- 📋 **Multiple Export Formats**: JSON and XMI output support
- 🐳 **Docker Ready**: Containerized deployment with CI/CD

## Architecture

### Core Components

- **NLP Pipeline**: spaCy + flashtext for fast text processing
- **Rule Engine**: JSON/YAML configurable rules for section detection, assertion, and negation
- **Agentic Layer**: LangChain agents for LLM-powered normalization
- **Active Learning**: Human-in-the-loop feedback system
- **Web Interface**: FastAPI + React for document upload and annotation review

### Data Flow

1. Document ingestion and preprocessing
2. spaCy tokenization and basic NLP
3. Flashtext dictionary matching
4. Rule-based section/assertion/negation detection
5. Agentic LLM routing for ambiguous cases
6. Structured output generation (JSON/XMI)
7. Active learning feedback loop

## Quick Start

```bash
# Clone the repository
git clone https://github.com/sonishsivarajkumar/Agentic-MedTagger.git
cd Agentic-MedTagger

# Install dependencies
pip install -r requirements.txt

# Set up the database
python scripts/setup_db.py

# Run the application
python -m agentic_medtagger.main
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## Community

Join our growing community of clinical NLP developers and researchers!
