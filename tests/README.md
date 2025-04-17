# Unit Tests for Causal Evaluation

This directory contains unit tests for the causal-eval project.

## Running Tests

There are multiple ways to run the tests:

### 1. Using the test runner script

The test runner script runs all tests and generates a coverage report:

```bash
./tests/run_tests.py
```

### 2. Using pytest directly

You can run all tests:

```bash
pytest -v
```

Or run specific test files:

```bash
pytest -v tests/test_strokesimulation.py
```

### 3. Using Python module

```bash
python -m pytest -v
```

## Coverage Report

The test runner script generates a coverage report that shows which parts of the code are tested. 
After running the tests with the runner script, you can view the HTML coverage report at:

```
coverage_html/index.html
```

## Structure

- `conftest.py`: Configuration and fixtures for pytest
- `test_*.py`: Test files for each module in the src directory
- `run_tests.py`: Script to run tests with coverage reporting