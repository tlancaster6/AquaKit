---
paths: ["src/**/*.py"]
---

## `__init__.py` Public API

- Every package must have an `__init__.py` that explicitly imports its public interface
- Define `__all__` listing all public names
- Include a module-level docstring describing the package's purpose

```python
"""Utility functions for data processing and validation."""

from .processing import clean_data, transform_records
from .validation import check_schema, validate_input

__all__ = ["clean_data", "transform_records", "check_schema", "validate_input"]
```

## Module-Level Docstrings

- Every `.py` file must start with a one-line docstring describing its purpose

## Public Functions

- All public functions must have type hints on parameters and return type
- Include parameter semantics and constraints in Google-style docstrings

## Updating `__init__.py`

- When adding a new public function or class, add it to the parent package's `__init__.py` and `__all__`
- When renaming or removing a public symbol, update `__init__.py` and `__all__` accordingly
