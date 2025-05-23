#!/usr/bin/env python
import argparse
import pickle
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

def load_simulation_results(file_path):
    """Load the simulation results pickle file and return the dataframe."""
    with open(file_path, 'rb') as f:
        results = pickle.load(f)
    # The dataframe is stored under the 'data' key
    df = results['data']
    return df

def prepare_data(df):
    """
    Process the dataframe to extract the month 0 risk score,
    define the treatment indicator, and compute the running variable.
    
    - month0_risk: first element from the risk_scores list.
    - treatment: 1 if intervention_applied is True and intervention_month==0.
    - running: month0_risk minus the cutoff risk (minimum risk among treated patients).
    """
    # Extract the first month risk score for each patient.
    df['month0_risk'] = df['risk_scores'].apply(lambda scores: scores[0] if isinstance(scores, list) and len(scores) > 0 else None)
    
    # Remove rows with missing risk scores.
    df = df[df['month0_risk'].notnull()].copy()
    
    # Define treatment: patients who received the intervention in month 0.
    df['treatment'] = df.apply(lambda row: 1 if row['intervention_applied'] and row['intervention_month'] == 0 else 0, axis=1)
    
    # Determine the cutoff risk: minimum risk score among those treated in month 0.
    treated = df[df['treatment'] == 1]
    if treated.empty:
        raise ValueError("No treated patients in month 0 were found!")
    cutoff = treated['month0_risk'].min()
    
    # Define the running variable: distance from the cutoff.
    df['running'] = df['month0_risk'] - cutoff
    return df, cutoff

def run_rdd_regression(df):
    """
    Fit a regression discontinuity model:
    
        had_stroke ~ treatment + running + treatment:running
    
    Using robust (heteroskedasticity-consistent) standard errors.
    """
    formula = 'had_stroke ~ treatment + running + treatment:running'
    model = smf.ols(formula, data=df).fit(cov_type='HC1')
    return model

def plot_rdd(df, cutoff):
    """
    Create a binned scatter plot of stroke incidence (had_stroke) versus the running variable.
    A vertical dashed line indicates the intervention threshold (i.e. running==0).
    """
    plt.figure(figsize=(8,6))
    # Create bins for the running variable.
    bins = np.linspace(df['running'].min(), df['running'].max(), 40)
    df['bin'] = pd.cut(df['running'], bins=bins)
    binned = df.groupby('bin')['had_stroke'].mean()
    
    # Compute bin centers.
    bin_centers = [interval.mid for interval in binned.index.categories]
    
    plt.scatter(bin_centers, binned, color='blue', label='Binned Stroke Rate')
    plt.axvline(x=0, color='red', linestyle='--', label='Intervention Cutoff')
    plt.xlabel('Month 0 Risk Score - Cutoff')
    plt.ylabel('Stroke Rate')
    plt.title('Regression Discontinuity Plot')
    plt.legend()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Run Regression Discontinuity Evaluation on Simulation Output")
    parser.add_argument("--input_file", type=str, required=True, 
                        help="Path to the simulation results pickle file (e.g., simulation_results_xxx.pkl)")
    args = parser.parse_args()
    
    # Load and process the simulation data.
    df = load_simulation_results(args.input_file)
    df, cutoff = prepare_data(df)
    print("Calculated cutoff risk score (month 0):", cutoff)
    
    # Run the RD regression.
    model = run_rdd_regression(df)
    print("\nRegression Discontinuity Model Summary:")
    print(model.summary())
    
    # Plot the binned scatter for visual inspection.
    plot_rdd(df, cutoff)

if __name__ == '__main__':
    main()
