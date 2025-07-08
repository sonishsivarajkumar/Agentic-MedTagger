# Contributing to Agentic MedTagger

Thank you for your interest in contributing to Agentic MedTagger! This document provides guidelines for contributing to the project.

## Getting Started

### Development Environment Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/sonishsivarajkumar/Agentic-MedTagger.git
   cd Agentic-MedTagger
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

5. **Set up the database**:
   ```bash
   python scripts/setup_db.py
   ```

6. **Download spaCy model**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=agentic_medtagger

# Run specific test file
pytest tests/test_core.py
```

### Code Style

We use the following tools for code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

Run all checks:
```bash
black src tests
isort src tests
flake8 src tests
mypy src
```

## Contributing Guidelines

### Types of Contributions

1. **Bug Reports**: Use GitHub issues with the "bug" label
2. **Feature Requests**: Use GitHub issues with the "enhancement" label
3. **Documentation**: Improvements to docs, README, or code comments
4. **Code Contributions**: Bug fixes, new features, performance improvements

### Pull Request Process

1. **Fork the repository** and create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the coding standards

3. **Add tests** for new functionality

4. **Update documentation** if needed

5. **Ensure all tests pass**:
   ```bash
   pytest
   ```

6. **Submit a pull request** with:
   - Clear description of changes
   - Reference to related issues
   - Screenshots for UI changes

### Coding Standards

#### Python Code

- Follow PEP 8 style guidelines
- Use type hints for all function signatures
- Write docstrings for all public functions and classes
- Keep functions small and focused (max ~50 lines)
- Use meaningful variable and function names

#### Documentation

- Use Google-style docstrings
- Include examples in docstrings where helpful
- Update README.md for significant changes
- Add inline comments for complex logic

#### Testing

- Write unit tests for all new functionality
- Aim for >90% test coverage
- Use descriptive test names
- Include integration tests for major features

### Clinical NLP Specific Guidelines

#### Medical Data Handling

- **Never commit real patient data**
- Use synthetic or de-identified data for examples
- Follow HIPAA guidelines in all code
- Include privacy considerations in documentation

#### Annotation Standards

- Follow standard clinical NLP annotation schemes when possible
- Document annotation guidelines clearly
- Include inter-annotator agreement metrics for new tasks
- Support standard formats (BRAT, XMI, JSON)

#### UMLS Integration

- Respect UMLS licensing terms
- Provide clear instructions for UMLS setup
- Support multiple UMLS versions
- Include concept mapping validation

### Adding New Components

#### Pipeline Components

1. Inherit from `PipelineComponent`
2. Implement `process()` method
3. Add configuration validation
4. Include comprehensive tests
5. Document configuration options

Example:
```python
class MyAnnotator(PipelineComponent):
    """Custom annotator for X task."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("my_annotator", config)
        # Initialize your component
    
    async def process(self, document: Document) -> Document:
        """Process document and add annotations."""
        # Your implementation
        return document
```

#### Agents

1. Use LangChain patterns
2. Support multiple LLM providers
3. Include confidence scoring
4. Add fallback mechanisms
5. Log all LLM interactions

#### Web Interface

1. Use React with TypeScript
2. Follow accessibility guidelines
3. Include responsive design
4. Add comprehensive error handling
5. Write component tests

### Performance Guidelines

- **Profile before optimizing**: Use cProfile or py-spy
- **Memory efficiency**: Monitor memory usage for large documents
- **Async processing**: Use async/await for I/O operations
- **Batch processing**: Support processing multiple documents
- **Caching**: Cache expensive operations (model loading, etc.)

### Security Guidelines

- **Input validation**: Validate all user inputs
- **SQL injection**: Use parameterized queries
- **File uploads**: Validate file types and sizes
- **API security**: Use proper authentication
- **Dependencies**: Keep dependencies updated

## Release Process

### Version Numbers

We use semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

### Release Checklist

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Run full test suite
4. Update documentation
5. Create GitHub release
6. Build and publish to PyPI

## Community

### Communication

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Documentation**: In-code docs and README

### Code of Conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/) code of conduct. Please be respectful and inclusive in all interactions.

### Recognition

Contributors will be recognized in:
- `CONTRIBUTORS.md` file
- GitHub release notes
- Documentation acknowledgments

## Getting Help

If you need help:

1. Check existing documentation
2. Search GitHub issues
3. Create a new issue with the "question" label
4. Join GitHub discussions

Thank you for contributing to Agentic MedTagger! 🏥🤖
