import unittest
import numpy as np
from src.sim_grid_search import simulate_run

class TestSimGridSearch(unittest.TestCase):
    def setUp(self):
        # Set a fixed seed for reproducibility
        np.random.seed(42)
    
    def test_simulate_run(self):
        # Test simulation with fixed parameters
        neg_alpha = 0.5
        neg_beta = 3.0
        pos_alpha = 3.0
        pos_beta = 0.5
        pop_size = 1000
        prevalence = 0.1
        top_threshold = 100
        
        # Run the simulation
        ppv, sensitivity, f1 = simulate_run(
            neg_alpha, neg_beta, pos_alpha, pos_beta, 
            pop_size, prevalence, top_threshold
        )
        
        # Check that all metrics are within valid ranges (0-1)
        self.assertGreaterEqual(ppv, 0)
        self.assertLessEqual(ppv, 1)
        self.assertGreaterEqual(sensitivity, 0)
        self.assertLessEqual(sensitivity, 1)
        self.assertGreaterEqual(f1, 0)
        self.assertLessEqual(f1, 1)
        
        # With these parameters, we expect reasonable performance
        # The exact values will depend on the random seed
        self.assertGreater(ppv, 0.2)
        self.assertGreater(sensitivity, 0.1)
    
    def test_edge_cases(self):
        # Very low prevalence
        ppv, sensitivity, f1 = simulate_run(
            0.5, 3.0, 3.0, 0.5, 1000, 0.01, 10
        )
        
        # High prevalence
        ppv2, sensitivity2, f2 = simulate_run(
            0.5, 3.0, 3.0, 0.5, 1000, 0.9, 900
        )
        
        # With high prevalence and larger intervention group:
        # - PPV can vary but is typically high
        # - Sensitivity should be higher compared to low prevalence with small intervention
        self.assertGreaterEqual(sensitivity2, 0.5)  # Expect reasonable sensitivity with high prevalence
        self.assertGreaterEqual(ppv2, 0.5)  # Expect reasonable PPV with high prevalence
    
    def test_beta_distributions(self):
        # Test that negative and positive scores follow expected distributions
        # For negative events (right-skewed), most scores should be lower
        # For positive events (left-skewed), most scores should be higher
        
        # Generate scores for 5000 negative and 5000 positive events
        neg_scores = np.random.beta(0.5, 3.0, 5000)
        pos_scores = np.random.beta(3.0, 0.5, 5000)
        
        # Calculate means - positive scores should have higher mean
        neg_mean = np.mean(neg_scores)
        pos_mean = np.mean(pos_scores)
        
        self.assertLess(neg_mean, 0.5)  # Negative mean should be < 0.5
        self.assertGreater(pos_mean, 0.5)  # Positive mean should be > 0.5
        self.assertLess(neg_mean, pos_mean)  # Positive mean should exceed negative mean

if __name__ == '__main__':
    unittest.main()