import unittest
import importlib

pd_spec = importlib.util.find_spec("pandas")
np_spec = importlib.util.find_spec("numpy")

if pd_spec is not None and np_spec is not None:
    import pandas as pd
    import numpy as np
    from src.reg_disc import prepare_data, run_rdd_regression
from unittest.mock import patch, mock_open, MagicMock

@unittest.skipUnless(pd_spec is not None and np_spec is not None, "pandas/numpy not available")
class TestRegDisc(unittest.TestCase):
    def setUp(self):
        # Create a test dataframe
        self.df = pd.DataFrame({
            'person_id': range(20),
            'risk_scores': [[0.8, 0.7], [0.75, 0.7], [0.7, 0.65], [0.65, 0.6], [0.6, 0.55],
                           [0.55, 0.5], [0.5, 0.45], [0.45, 0.4], [0.4, 0.35], [0.35, 0.3],
                           [0.3, 0.25], [0.25, 0.2], [0.2, 0.15], [0.15, 0.1], [0.1, 0.05],
                           [0.05, 0.04], [0.04, 0.03], [0.03, 0.02], [0.02, 0.01], [0.01, 0.005]],
            'intervention_applied': [True, True, True, True, True, 
                                    True, True, True, True, True,
                                    False, False, False, False, False,
                                    False, False, False, False, False],
            'intervention_month': [0, 0, 0, 0, 0, 
                                  0, 0, 0, 0, 0,
                                  None, None, None, None, None,
                                  None, None, None, None, None],
            'had_stroke': [0, 0, 0, 1, 0,
                          0, 1, 0, 0, 1,
                          1, 1, 1, 0, 0,
                          1, 0, 0, 1, 0]
        })
    
    def test_prepare_data(self):
        df, cutoff = prepare_data(self.df)
        
        # Check that month0_risk was extracted correctly
        self.assertEqual(df.loc[0, 'month0_risk'], 0.8)
        self.assertEqual(df.loc[19, 'month0_risk'], 0.01)
        
        # Check that treatment was defined correctly
        self.assertEqual(df.loc[0, 'treatment'], 1)
        self.assertEqual(df.loc[10, 'treatment'], 0)
        
        # Check that the cutoff is the minimum risk score among treated patients
        self.assertEqual(cutoff, 0.35)
        
        # Check that running variable was calculated correctly
        self.assertAlmostEqual(df.loc[0, 'running'], 0.45, places=10)  # 0.8 - 0.35
        self.assertAlmostEqual(df.loc[10, 'running'], -0.05, places=10)  # 0.3 - 0.35
    
    def test_run_rdd_regression(self):
        # First prepare the data
        df, _ = prepare_data(self.df)
        
        # Run the regression
        model = run_rdd_regression(df)
        
        # Check that we got a model back
        self.assertIsNotNone(model)
        
        # Check that the model has the expected variables
        expected_vars = ['Intercept', 'treatment', 'running', 'treatment:running']
        self.assertListEqual(list(model.params.index), expected_vars)
        
        # Check that the model has reasonable p-values (either significant or not)
        for p_value in model.pvalues:
            self.assertGreaterEqual(p_value, 0)
            self.assertLessEqual(p_value, 1)

if __name__ == '__main__':
    unittest.main()
