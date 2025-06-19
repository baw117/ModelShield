# Contributing to ModelShield

Thank you for your interest in contributing to ModelShield! 🛡️ This project relies on community contributions to stay effective against evolving AI safety challenges.

## 🌟 How to Contribute

### Types of Contributions We Need

#### 🔍 Detection Patterns
- **New Attack Vectors**: Help identify emerging prompt injection techniques
- **Pattern Improvements**: Enhance existing regex patterns for better accuracy
- **False Positive Reduction**: Improve context-aware detection logic

#### 🧠 ML Models
- **Model Integration**: Add new transformer models for semantic detection
- **Performance Optimization**: Improve model inference speed
- **Multi-language Support**: Extend detection to non-English languages

#### 🔧 Framework Enhancements
- **New Features**: Add authentication, rate limiting, or monitoring features
- **Performance**: Optimize for higher throughput and lower latency
- **Documentation**: Improve guides, examples, and API documentation

#### 🧪 Testing & Validation
- **Test Cases**: Add comprehensive test scenarios
- **Benchmarking**: Create evaluation datasets
- **Edge Cases**: Identify and handle corner cases

## 📋 Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/baw117/ModelShield.git
cd modelshield
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies

# Install spaCy model
python -m spacy download en_core_web_sm
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

## 🛠️ Development Guidelines

### Code Style

- **Python**: Follow PEP 8 guidelines
- **Formatting**: Use `black` for code formatting
- **Linting**: Use `flake8` for linting
- **Type Hints**: Include type hints for new functions

```bash
# Format code
black .

# Run linting
flake8 .

# Run type checking
mypy modelshield/
```

### Testing

All contributions must include appropriate tests:

```bash
# Run all tests
python -m pytest

# Run specific test file
python test_server.py

# Run with coverage
pytest --cov=modelshield tests/
```

### Documentation

- Update README.md if adding new features
- Include docstrings for new functions/classes
- Add inline comments for complex logic
- Update API documentation if changing endpoints

## 🔍 Adding New Detection Patterns

### Pattern Categories

When adding new patterns, categorize them appropriately:

```python
# In guardrails_server.py, add to appropriate category:
def _get_your_category_patterns(self) -> Dict[str, List[str]]:
    return {
        "subcategory_name": [
            r'(?i)your_regex_pattern_here',
            r'(?i)another_pattern',
        ],
        "another_subcategory": [
            r'(?i)more_patterns',
        ]
    }
```

### Pattern Guidelines

1. **Use case-insensitive patterns**: Start with `(?i)`
2. **Be specific**: Avoid overly broad patterns that cause false positives
3. **Test thoroughly**: Include test cases for new patterns
4. **Document intent**: Add comments explaining what the pattern detects

Example:
```python
# Good pattern - specific and well-documented
r'(?i)\bhow\s+to\s+(bypass|circumvent|avoid)\s+.*(security|filter|detection)',

# Avoid - too broad, could match legitimate content
r'(?i)\bbypass\b',
```

## 🧪 Testing Your Contributions

### 1. Unit Tests

```bash
# Test specific components
python pii_detector.py
python semantic_detector.py
python output_scanner.py
```

### 2. Integration Tests

```bash
# Start the server
python startup.py

# In another terminal, run integration tests
python test_server.py
```

### 3. Evaluation Framework

```bash
# Run comprehensive evaluation
python -c "
from comprehensive_test_guardrails import main
import asyncio
asyncio.run(main())
"
```

## 📝 Pull Request Process

### 1. Before Submitting

- [ ] Code follows project style guidelines
- [ ] All tests pass locally
- [ ] New features include tests
- [ ] Documentation is updated
- [ ] No security-sensitive information is exposed

### 2. Pull Request Description

Include in your PR description:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No sensitive data exposed
```

### 3. Review Process

1. **Automated Checks**: CI/CD pipeline runs tests
2. **Code Review**: Maintainers review changes
3. **Testing**: Additional testing if needed
4. **Merge**: Approved changes are merged

## 🔒 Security Considerations

### What NOT to Include

- **Real attack payloads**: Use obfuscated or sanitized examples
- **Specific vulnerability details**: Follow responsible disclosure
- **Production credentials**: Never commit real API keys or passwords
- **Detailed exploit instructions**: Focus on detection, not exploitation

### Safe Pattern Development

```python
# Good - Pattern that detects without explaining how to exploit
r'(?i)\b(bypass|circumvent)\s+.*(security|filter)',

# Avoid - Specific exploit instructions
# Don't include actual working exploit code
```

## 📚 Resources

### Learning Materials

- [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [Microsoft Presidio Documentation](https://microsoft.github.io/presidio/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

### Development Tools

- **Code Editor**: VS Code with Python extension
- **Testing**: pytest, pytest-cov
- **Documentation**: Sphinx (for advanced docs)
- **Profiling**: cProfile, line_profiler

## 🤝 Community

### Getting Help

- **GitHub Discussions**: For questions and ideas
- **GitHub Issues**: For bug reports and feature requests
- **Code Review**: Learn from pull request feedback

### Communication Guidelines

- Be respectful and inclusive
- Focus on constructive feedback
- Ask questions if something is unclear
- Help newcomers get started

## 🎯 Priority Areas

We especially need help with:

1. **Multi-language Detection**: Patterns for non-English content
2. **Performance Optimization**: Making detection faster
3. **Edge Case Handling**: Unusual input scenarios
4. **Documentation**: Better examples and guides
5. **Integration Examples**: Usage with popular frameworks

## 📊 Recognition

Contributors will be:
- Listed in the README.md contributors section
- Mentioned in release notes for significant contributions
- Invited to join the core contributor team for ongoing contributions

---

**Thank you for helping make AI systems safer! 🛡️**

*Every contribution, no matter how small, helps protect users and improves AI safety for everyone.*