# Guidelines for AI Agents

This repository contains Python code and Jupyter notebooks for simulating and evaluating healthcare interventions. Use this guide when contributing automated changes.

## Project Structure
- `src/` – Core Python modules for simulations and analysis.
- `notebooks/` – Jupyter notebooks used for data exploration and method experimentation. They may also contain embedded tests via `ipytest`.
- `tests/` – Unit tests covering the modules in `src/`. Includes a `run_tests.py` helper for running tests with coverage.
- Other files such as `config.yaml`, `requirements.txt`, and `requirements-notebook.txt` contain configuration and dependencies.

## Coding Conventions
- **Language**: Python 3.9.
- **Libraries**: numpy, pandas, matplotlib, statsmodels and related scientific libraries. Jupyter notebooks rely on the packages listed in `requirements-notebook.txt`.
- **Style**: Follow standard PEP8 conventions. Use snake_case for functions and variables and PascalCase for classes. Provide descriptive docstrings and type hints where practical.

## Testing Protocols
- Unit tests live in the `tests/` directory and are executed with `pytest`.
- Run the full suite with coverage using:
  ```bash
  python tests/run_tests.py
  ```
  or simply:
  ```bash
  pytest -v
  ```
- Notebooks may embed tests using `ipytest`. To run them inside a notebook:
  ```python
  import ipytest
  ipytest.run('-q')
  ```
- Write meaningful tests when modifying or adding functionality. Aim for good coverage and clear assertions.

## Pull Request Guidelines
- Provide a clear summary of changes and reference any related issues.
- Ensure all tests pass and coverage reports generate without errors.
- Keep PRs focused: avoid mixing unrelated changes.
- Use the repository’s GitHub Actions (`Python Tests` and `Run Tests`) as confirmation that checks pass.

## Programmatic Checks
Run these commands before opening a PR:
```bash
flake8 src tests            # style check (if flake8 is installed)
mypy src                    # optional static type checking
python tests/run_tests.py   # run unit tests with coverage
```

When notebooks contain tests via `ipytest`, execute the notebook cells to ensure those tests succeed as well.
