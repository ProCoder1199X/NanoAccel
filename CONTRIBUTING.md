# Contributing to NanoAccel

Thank you for your interest in contributing to NanoAccel! This document provides guidelines for contributing to the project.

## Getting Started

### Development Setup

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/NanoAccel.git
   cd NanoAccel
   ```
3. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```
4. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

### Development Environment

- **Python**: 3.8 or higher
- **Code formatting**: Black (line length 88)
- **Linting**: Flake8
- **Type checking**: MyPy
- **Testing**: Pytest with coverage

## Contribution Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use Black for code formatting
- Ensure type hints are provided for all functions
- Write comprehensive docstrings

### Testing

- Write tests for all new functionality
- Maintain test coverage above 80%
- Use descriptive test names
- Include both unit and integration tests

### Documentation

- Update docstrings for any new functions/classes
- Update README.md if adding new features
- Add examples for new functionality
- Update type hints as needed

## Pull Request Process

1. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the guidelines above

3. **Run tests** to ensure everything works:
   ```bash
   pytest
   ```

4. **Check code quality**:
   ```bash
   black nanoaccel/ tests/
   flake8 nanoaccel/ tests/
   mypy nanoaccel/
   ```

5. **Commit your changes** with a descriptive message:
   ```bash
   git commit -m "Add feature: brief description"
   ```

6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request** on GitHub

### Pull Request Requirements

- Clear description of changes
- Reference any related issues
- Include tests for new functionality
- Update documentation as needed
- Ensure all CI checks pass

## Types of Contributions

### Bug Reports

When reporting bugs, please include:

- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version, etc.)
- Error messages or logs

### Feature Requests

For feature requests, please provide:

- Clear description of the feature
- Use case and motivation
- Proposed implementation (if applicable)
- Any relevant research or references

### Code Contributions

We welcome contributions in these areas:

- **Performance optimizations**
- **New quantization methods**
- **Additional model support**
- **Improved speculative decoding**
- **Better CPU optimization**
- **Enhanced CLI features**
- **Documentation improvements**
- **Test coverage improvements**

## Development Guidelines

### Architecture

- **Modular design**: Keep components loosely coupled
- **Error handling**: Provide meaningful error messages
- **Logging**: Use appropriate log levels
- **Configuration**: Support both file and environment variable configuration

### Performance

- **Benchmarking**: Include performance tests for optimizations
- **Memory usage**: Monitor and optimize memory consumption
- **CPU efficiency**: Ensure optimizations work across different CPU types

### Compatibility

- **Python versions**: Support Python 3.8+
- **Operating systems**: Test on Windows, Linux, and macOS
- **Hardware**: Consider low-end hardware constraints

## Code Review Process

- All PRs require review from maintainers
- Address feedback promptly
- Be open to suggestions and improvements
- Maintain a collaborative and respectful tone

## Release Process

- Version numbers follow [Semantic Versioning](https://semver.org/)
- Releases are created by maintainers
- Changelog is updated for each release

## Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: Contact maintainers for sensitive issues

## License

By contributing to NanoAccel, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes for significant contributions
- Project documentation

Thank you for contributing to NanoAccel! 🚀
