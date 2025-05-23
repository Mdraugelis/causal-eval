# Causal Evaluation Framework for Healthcare Interventions

**A Python-based framework for simulating healthcare interventions and evaluating their causal effects.** This repository contains two main applications:

1. **Stroke Simulation Framework**: Simulate stroke events in a population over time, with or without interventions.
2. **Causal Analysis Tools**: Generate synthetic data and apply causal inference methods to evaluate intervention effectiveness.

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](tests/run_tests.py)

---

## Table of Contents

1. [Applications Overview](#applications-overview)
2. [Installation](#installation)
3. [Usage](#usage)
   - [Stroke Simulation](#stroke-simulation)
   - [Causal Analysis](#causal-analysis)
4. [Testing](#testing)
5. [Features](#features)
6. [Configuration](#configuration)

---

## Applications Overview

### 1. Stroke Simulation Framework

The stroke simulation framework models stroke events in a population over time. It includes:

- **Base Simulation** (`strokesimulation.py`): Simulate a population with configurable stroke incidence and mortality rates.
- **Intervention Simulation** (`intervention_sim.py`): Enhanced simulation that allows for monthly targeted interventions based on risk scores.

Both simulations track individual outcomes over time, generate risk scores for each person, and enable evaluation of different intervention strategies.

### 2. Causal Analysis Tools

The causal analysis tools provide methods for evaluating the causal effect of interventions:

- **Regression Discontinuity Analysis** (`reg_disc.py`): Implement RD design to evaluate intervention effectiveness at a threshold.
- **Synthetic Data Generation** (`claude_run.py`): Generate synthetic populations with controlled characteristics for causal evaluation.
- **Parameter Optimization** (`sim_grid_search.py`): Find optimal beta distribution parameters for simulating risk scores.

These tools enable researchers to test causal inference methods in controlled environments where ground truth is known.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Mdraugelis/causal-eval.git
   cd causal-eval
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the core dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. *(Optional)* Install additional packages for notebook analysis:
   ```bash
   pip install -r requirements-notebook.txt
   ```

---

## Usage

### Stroke Simulation

#### Basic Simulation

Run a basic stroke simulation without interventions:

```bash
python -c "from src.strokesimulation import main; main()"
```

This will use the parameters defined in `config.yaml` to run the simulation.

#### Intervention Simulation

Run a simulation with monthly targeted interventions:

```bash
python -c "from src.intervention_sim import main; main()"
```

This simulation selects high-risk individuals each month for intervention and reduces their stroke probability.

### Causal Analysis

#### Regression Discontinuity Analysis

To run regression discontinuity analysis on simulation results:

```bash
python src/reg_disc.py --input_file results/simulation_results_XXXX.pkl
```

Replace `XXXX` with the simulation ID from a previous run.

#### Synthetic Data and RD Simulation

Generate synthetic data and run an RD analysis:

```python
from src.claude_run import run_simulation_rd

results = run_simulation_rd(
    pop_size=50_000,
    baseline_prevalence=0.05,
    top_k_factor=2,
    intervention_efficacy=0.3,
    top_k=250,
    months=12,
    bandwidth=0.02,
    random_seed=42
)
```

#### Parameter Grid Search

Optimize beta distribution parameters for risk score generation:

```bash
python src/sim_grid_search.py
```

Follow the interactive prompts to specify target PPV, sensitivity, and other parameters.

---

## Testing

Run the test suite with coverage reporting:

```bash
python tests/run_tests.py
```

This will generate a test report and create a coverage report in `coverage_html/index.html`.

---

## Features

- **Configurable Population**: Specify population size, age distribution, and baseline risk
- **Risk Modeling**: Generate risk scores with configurable distributions
- **Intervention Strategies**: Test different selection criteria and effectiveness levels
- **Causal Inference**: Implement and validate RD designs
- **Detailed Logging**: Track outcomes at individual and population levels
- **Visualization**: Generate plots for RD analysis and risk distributions

---

## Configuration

The simulation parameters can be configured in `config.yaml`:

```yaml
# Example configuration
population_size: 40000
annual_incidence_rate: 0.05
num_years: 2
include_mortality: true
annual_mortality_rate: 0.01
age_distribution: true
initial_age_range: [30, 70]
```

For the intervention simulation, additional parameters can be specified:

```python
intervention_effectiveness: 0.2  # 20% reduction in stroke risk
num_interventions: 250          # Top 250 patients selected monthly
```

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## License

This project is open source and available under the MIT License.
