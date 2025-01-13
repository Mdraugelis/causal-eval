# Stroke Incidence & Intervention Simulation

**A Python-based simulation framework for exploring stroke incidence, machine learning–based risk models, and intervention strategies.** This repository offers:

- **Stroke Population Simulation**: Create synthetic populations with configurable incidence and risk factors.
- **Machine Learning Risk Score Simulator**: Assign risk scores to each individual, matching desired sensitivity, specificity, and PPV.
- **Intervention Protocol Simulator**: Model targeted interventions (e.g., monthly ranking of high-risk individuals) and measure impact with causal inference methods such as **Difference-in-Differences (DiD)** and **Regression Discontinuity (RD)**.

---

## Table of Contents

1. [Overview](#overview)  
2. [Key Features](#key-features)   
5. [Simulation Workflow](#simulation-workflow)  
6. [Causal Inference Modules](#causal-inference-modules)  


---

## Overview

Healthcare researchers, data scientists, and public health analysts often need **synthetic data** to test ideas, develop machine learning models, or evaluate the efficacy of interventions. This repository provides a simulation environment for:

1. **Generating a synthetic population** with user-defined sizes, stroke prevalence, and risk distributions.
2. **Assigning risk scores** to individuals using machine learning model assumptions (e.g., specify target sensitivity, specificity, PPV).
3. **Applying interventions**, such as selecting top-risk patients monthly for a risk reduction protocol, and measuring outcomes.

You can **compare** different intervention strategies or evaluate the performance of causal inference methods (RD, DiD) under controlled, simulated conditions.

---

## Key Features

- **Configurable Population**  
  - Specify population size, age distribution, comorbidities, and mortality assumptions.
- **Monthly Stroke Incidence Model**  
  - Convert annual stroke incidence to monthly probabilities, or use a survival/hazard function.
  - Log stroke events by individual and month.
- **Machine Learning Risk Score Simulator**  
  - Assign risk scores that reflect a chosen sensitivity, specificity, and positive predictive value.
  - Flexible threshold-based labeling.
- **Intervention Protocol**  
  - Rank patients each month by their risk.
  - Assign an intervention to the top X (e.g., 250) or anyone above a certain threshold.
  - Reduce stroke risk by a configurable percentage.
- **Causal Inference Evaluation**  
  - **Difference-in-Differences (DiD)**: Define treatment/control groups with pre-/post-intervention periods.
  - **Regression Discontinuity (RD)**: Implement a cutoff-based intervention assignment to test RD assumptions.
- **Detailed Logging & Reporting**  
  - Monthly incidence, cumulative stroke counts, confusion matrices for ML models, etc.

## Simulation Workflow
1. Population Initialization
- Create N individuals with baseline risk factors, ages, and IDs.
2. Risk Model (Optional)
- Assign or compute monthly stroke probabilities.
- If using a machine learning approach, create or load a model to generate risk scores.
3. Monthly Loop
- Update each person’s risk (e.g., increment age, apply hazard function).
- Rank individuals (top 250 or threshold-based).
- Intervention: Reduce risk for selected individuals by XX%.
- Simulate stroke events.
- Log outcomes.
4. Final Reporting
- Cumulative strokes, survival curves, incidence rates.
- Compare intervention vs. non-intervention.
- Summaries of model performance (sensitivity, specificity, PPV).

## Causal Inference Modules
Difference-in-Differences (DiD)
- Pre/Post Periods: Run the simulation with a baseline (no intervention) phase and an intervention phase.
- Treatment vs. Control: Randomly assign part of the population to treatment at a specific time.
- Analysis: Estimate DiD effect by comparing changes in stroke incidence between groups over time.

Regression Discontinuity (RD)
- Cutoff: Define a stable threshold on the risk score to assign treatment vs. non-treatment.
- Local Randomization: Explore outcomes for individuals near the threshold.
- Analysis: Estimate RD effect and compare to known, simulated ground truth.
