# Contributing to SSN

## Development Setup

```bash
git clone https://github.com/yourusername/ssn.git
cd ssn
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest
pytest --cov=ssn  # with coverage
```

## Code Style

- Black for formatting
- Ruff for linting
- Type hints where practical

```bash
black ssn tests
ruff check ssn tests
mypy ssn
```

## Pull Request Process

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-schema`)
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit PR with clear description

## Adding New Schemas

Add domain-specific schemas in `ssn/core.py`:

```python
def _register_defaults(self):
    # ... existing schemas ...
    
    self.register_schema(
        "your_schema",
        "full_action_name",
        ["arg1", "arg2"],
        {"default_key": "default_value"}
    )
```

## Reporting Issues

Include:
- Python version
- SSN version
- Minimal reproducible example
- Expected vs actual behavior
