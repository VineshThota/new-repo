# Contributing to AI Slack Focus Assistant

We welcome contributions to the AI Slack Focus Assistant project! This document provides guidelines for contributing to the project.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Contribution Process](#contribution-process)
- [Testing](#testing)
- [Documentation](#documentation)
- [Community Guidelines](#community-guidelines)

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of AI/ML concepts
- Familiarity with Streamlit and pandas

### Areas for Contribution

We welcome contributions in the following areas:

1. **AI/ML Improvements**
   - Enhanced message classification algorithms
   - Better summarization techniques
   - Advanced NLP models integration
   - Performance optimizations

2. **Feature Development**
   - New visualization components
   - Additional analytics features
   - Integration with other platforms
   - Mobile-responsive design

3. **Bug Fixes**
   - UI/UX improvements
   - Performance issues
   - Edge case handling
   - Error handling enhancements

4. **Documentation**
   - Code documentation
   - User guides
   - API documentation
   - Tutorial content

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/new-repo.git
cd new-repo/ai-slack-focus-assistant
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy
```

### 4. Run the Application

```bash
# Start the Streamlit app
streamlit run app.py
```

### 5. Docker Setup (Optional)

```bash
# Build and run with Docker
docker-compose up --build
```

## Coding Standards

### Python Style Guide

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
- Use [Black](https://black.readthedocs.io/) for code formatting
- Use [flake8](https://flake8.pycqa.org/) for linting
- Use type hints where appropriate

### Code Formatting

```bash
# Format code with Black
black .

# Check linting with flake8
flake8 .

# Type checking with mypy
mypy .
```

### Naming Conventions

- **Functions and variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private methods**: `_leading_underscore`

### Documentation Standards

- Use docstrings for all functions, classes, and modules
- Follow [Google docstring format](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Include type hints in function signatures
- Add inline comments for complex logic

Example:

```python
def classify_priority(self, message: str, sender: str = None) -> Dict[str, Any]:
    """Classify message priority using multiple signals.
    
    Args:
        message: The message content to classify
        sender: Optional sender information for authority scoring
        
    Returns:
        Dictionary containing priority level, confidence score, and reasoning
        
    Raises:
        ValueError: If message is empty or invalid
    """
    # Implementation here
```

## Contribution Process

### 1. Create an Issue

Before starting work, create an issue to discuss:
- Bug reports with reproduction steps
- Feature requests with detailed descriptions
- Questions about implementation

### 2. Create a Branch

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/bug-description
```

### 3. Make Changes

- Write clean, well-documented code
- Follow the coding standards above
- Add tests for new functionality
- Update documentation as needed

### 4. Test Your Changes

```bash
# Run tests
pytest

# Run the app locally
streamlit run app.py

# Test with different scenarios
# - Various message types
# - Edge cases
# - Performance with large datasets
```

### 5. Commit Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: add advanced sentiment analysis for message classification"
```

#### Commit Message Format

Use conventional commit format:

- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Test additions or modifications
- `chore:` Maintenance tasks

### 6. Push and Create Pull Request

```bash
# Push to your fork
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- Clear title and description
- Reference to related issues
- Screenshots for UI changes
- Testing instructions

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=slack_ai_assistant

# Run specific test file
pytest tests/test_classifier.py
```

### Writing Tests

- Write unit tests for all new functions
- Include edge cases and error conditions
- Use descriptive test names
- Mock external dependencies

Example test:

```python
def test_message_classifier_urgent_detection():
    """Test that urgent messages are correctly classified."""
    classifier = MessageClassifier()
    urgent_message = "URGENT: Production server is down!"
    
    result = classifier.classify_priority(urgent_message)
    
    assert result['level'] == 'URGENT'
    assert result['confidence'] > 0.8
    assert 'urgency' in result['reasoning'].lower()
```

## Documentation

### Code Documentation

- Document all public APIs
- Include usage examples
- Explain complex algorithms
- Update README for new features

### User Documentation

- Update user guides for new features
- Include screenshots for UI changes
- Provide clear setup instructions
- Add troubleshooting sections

## Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Help newcomers get started
- Focus on the technical merits

### Communication

- Use GitHub issues for bug reports and feature requests
- Join discussions in pull requests
- Ask questions if you're unsure
- Share your ideas and suggestions

### Review Process

- All contributions require code review
- Address feedback promptly
- Be open to suggestions and improvements
- Maintain a collaborative attitude

## Getting Help

- **Issues**: Create a GitHub issue for bugs or questions
- **Discussions**: Use GitHub Discussions for general questions
- **Email**: Contact the maintainers for sensitive issues

## Recognition

Contributors will be:
- Listed in the project's contributors section
- Mentioned in release notes for significant contributions
- Invited to join the core team for sustained contributions

## License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to AI Slack Focus Assistant! 🚀