import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist
import logging

# Configure logging.
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -----------------------------------------------------------------------------
# Vectorized simulation run function.
# -----------------------------------------------------------------------------
def simulate_run(neg_alpha, neg_beta, pos_alpha, pos_beta, pop_size, prevalence, top_threshold):
    """
    Simulate one run in a fully vectorized manner.
    
    - Generates a population with a given prevalence.
    - Samples scores from Beta distributions (negative events: right-skewed; positive events: left-skewed).
    - Sorts events by score and marks the top 'top_threshold' as predicted positive.
    - Returns PPV, sensitivity, and F1 score.
    """
    # Determine the number of positive and negative events.
    num_pos = int(pop_size * prevalence)
    num_neg = pop_size - num_pos

    # Create and shuffle true labels.
    labels = np.concatenate((np.ones(num_pos, dtype=int), np.zeros(num_neg, dtype=int)))
    perm = np.random.permutation(pop_size)
    true_labels = labels[perm]

    # Vectorized sampling of scores.
    scores = np.empty(pop_size)
    pos_mask = (true_labels == 1)
    neg_mask = (true_labels == 0)
    scores[pos_mask] = np.random.beta(pos_alpha, pos_beta, size=np.count_nonzero(pos_mask))
    scores[neg_mask] = np.random.beta(neg_alpha, neg_beta, size=np.count_nonzero(neg_mask))
    
    # Sort scores in descending order.
    sorted_indices = np.argsort(scores)[::-1]
    
    # Label the top 'top_threshold' events as predicted positive.
    predictions = np.zeros(pop_size, dtype=int)
    predictions[sorted_indices[:top_threshold]] = 1

    # Compute confusion matrix elements.
    true_positives  = np.sum((predictions == 1) & (true_labels == 1))
    false_positives = np.sum((predictions == 1) & (true_labels == 0))
    false_negatives = np.sum((predictions == 0) & (true_labels == 1))
    
    # Calculate performance metrics.
    ppv = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = (2 * ppv * sensitivity / (ppv + sensitivity)) if (ppv + sensitivity) > 0 else 0

    return ppv, sensitivity, f1

# -----------------------------------------------------------------------------
# Grid search function that stops early when either:
#   (a) The candidate's PPV and Sensitivity are within ±0.02 of target values, or
#   (b) A maximum number of grid search iterations is reached.
# -----------------------------------------------------------------------------
def grid_search(target_ppv, target_sensitivity, pop_size, prevalence, top_threshold, num_runs, max_grid_iterations):
    # Define candidate ranges.
    neg_alpha_range = np.linspace(0.1, 0.9, 9)  # For right-skewed (must be <1)
    neg_beta_range  = np.linspace(1.1, 5, 10)     # For right-skewed (must be >1)
    pos_alpha_range = np.linspace(1.1, 5, 10)       # For left-skewed (must be >1)
    pos_beta_range  = np.linspace(0.1, 0.9, 9)      # For left-skewed (must be <1)

    best_error = np.inf
    best_params = None
    best_metrics = None
    convergence_data = []
    best_error_over_iterations = []
    iteration = 0
    stop_search = False  # Flag for early stopping

    total_iterations = len(neg_alpha_range) * len(neg_beta_range) * len(pos_alpha_range) * len(pos_beta_range)
    logging.info(f"Starting grid search over {total_iterations} combinations...")

    # Loop over all candidate parameter combinations.
    for na in neg_alpha_range:
        if stop_search:
            break
        for nb in neg_beta_range:
            if stop_search:
                break
            for pa in pos_alpha_range:
                if stop_search:
                    break
                for pb in pos_beta_range:
                    iteration += 1
                    
                    # Check maximum grid search iterations.
                    if iteration > max_grid_iterations:
                        logging.info("Maximum grid search iterations reached.")
                        stop_search = True
                        break

                    # Enforce parameter constraints.
                    if na >= 1 or nb <= 1:
                        continue
                    if pa <= 1 or pb >= 1:
                        continue

                    # Run the simulation 'num_runs' times for this candidate.
                    results = np.array([simulate_run(na, nb, pa, pb, pop_size, prevalence, top_threshold)
                                        for _ in range(num_runs)])
                    avg_ppv = np.mean(results[:, 0])
                    avg_sens = np.mean(results[:, 1])
                    avg_f1 = np.mean(results[:, 2])

                    # Calculate error.
                    error = abs(avg_ppv - target_ppv) + abs(avg_sens - target_sensitivity)
                    convergence_data.append(((na, nb, pa, pb), avg_ppv, avg_sens, avg_f1, error))

                    # Update best candidate if needed.
                    if error < best_error:
                        best_error = error
                        best_params = (na, nb, pa, pb)
                        best_metrics = (avg_ppv, avg_sens, avg_f1)
                        logging.info(
                            f"Iteration {iteration}/{total_iterations}: New best candidate - "
                            f"(na={na:.3f}, nb={nb:.3f}, pa={pa:.3f}, pb={pb:.3f}) | "
                            f"PPV: {avg_ppv:.3f}, Sensitivity: {avg_sens:.3f}, Error: {best_error:.3f}"
                        )

                    best_error_over_iterations.append(best_error)

                    # Check if the candidate meets the tolerance conditions.
                    if (abs(avg_ppv - target_ppv) <= 0.02) and (abs(avg_sens - target_sensitivity) <= 0.02):
                        logging.info(f"Candidate meets target tolerance at iteration {iteration}.")
                        stop_search = True
                        break  # Break out of innermost loop.
    
    return best_params, best_metrics, convergence_data, best_error_over_iterations

# -----------------------------------------------------------------------------
# Main function: collects user inputs, runs the grid search with early stopping,
# and plots convergence and final Beta distributions.
# -----------------------------------------------------------------------------
def main():
    # User inputs.
    pop_size = int(input("Enter the population size (integer): "))
    prevalence = float(input("Enter the mean prevalence of the positive event (0.0 to 1.0): "))
    num_runs = int(input("Enter the number of experimental runs per candidate: "))
    top_threshold = int(input("Enter the number of top risk scores to label as positive: "))
    target_ppv = float(input("Enter the target PPV (0.0 to 1.0): "))
    target_sensitivity = float(input("Enter the target Sensitivity (0.0 to 1.0): "))
    max_grid_iterations = int(input("Enter the maximum number of grid search iterations: "))

    # Run grid search.
    logging.info("Starting grid search on Beta distribution parameters...")
    best_params, best_metrics, convergence_data, best_error_over_iterations = grid_search(
        target_ppv, target_sensitivity, pop_size, prevalence, top_threshold, num_runs, max_grid_iterations
    )
    if best_params is None:
        logging.error("No valid parameter combination found. Please adjust grid ranges or inputs.")
        return

    neg_alpha_best, neg_beta_best, pos_alpha_best, pos_beta_best = best_params
    best_ppv, best_sens, best_f1 = best_metrics

    logging.info("Grid search complete.")
    logging.info(f"Best parameters found:")
    logging.info(f"  Negative events (right-skew): α = {neg_alpha_best:.3f}, β = {neg_beta_best:.3f}")
    logging.info(f"  Positive events (left-skew): α = {pos_alpha_best:.3f}, β = {pos_beta_best:.3f}")
    logging.info(f"Achieved Metrics: PPV = {best_ppv:.3f}, Sensitivity = {best_sens:.3f}, F1 = {best_f1:.3f}")

    # Plot convergence of grid search.
    plt.figure(figsize=(8, 5))
    plt.plot(best_error_over_iterations, marker='o', linestyle='-', color='b')
    plt.xlabel("Grid Search Iteration")
    plt.ylabel("Best Error so far")
    plt.title("Convergence of Grid Search")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot final Beta distributions.
    x = np.linspace(0, 1, 500)
    pdf_neg = beta_dist.pdf(x, neg_alpha_best, neg_beta_best)
    pdf_pos = beta_dist.pdf(x, pos_alpha_best, pos_beta_best)

    plt.figure(figsize=(8, 5))
    plt.plot(x, pdf_neg, label="Negative events (right-skew)", color='blue')
    plt.plot(x, pdf_pos, label="Positive events (left-skew)", color='red')
    plt.xlabel("Score")
    plt.ylabel("Probability Density")
    plt.title("Final Beta Distributions")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------
# Execute the main function.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
