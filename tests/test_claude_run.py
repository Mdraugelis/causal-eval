import unittest
import importlib
from unittest.mock import patch, MagicMock

pd_spec = importlib.util.find_spec("pandas")
np_spec = importlib.util.find_spec("numpy")

if pd_spec is not None and np_spec is not None:
    import pandas as pd
    import numpy as np
    from src.claude_run import (
        generate_population,
        calibrate_model_score,
        simulate_monthly_intervention,
        run_rd_analysis,
        run_simulation_rd,
    )


@unittest.skipUnless(pd_spec is not None and np_spec is not None, "pandas/numpy not available")
class TestClaudeRun(unittest.TestCase):
    def setUp(self):
        # Set fixed random seed for reproducibility
        np.random.seed(42)

        # Create a small synthetic population for testing
        self.population_size = 1000
        self.df = pd.DataFrame(
            {
                "patient_id": range(self.population_size),
                "true_risk": np.random.beta(a=2, b=38, size=self.population_size),
            }
        )
        # Scale true_risk to target prevalence
        target_prevalence = 0.05
        self.df["true_risk"] = self.df["true_risk"] * (target_prevalence / self.df["true_risk"].mean())

        # Add a predicted score (deterministic for testing)
        self.df["pred_score"] = self.df["true_risk"] + np.random.normal(0, 0.01, size=self.population_size)
        self.df["pred_score"] = np.clip(self.df["pred_score"], 0, 1)

    @patch("builtins.print")
    def test_generate_population(self, mock_print):
        # Test that population generation works
        pop_size = 1000
        baseline_prevalence = 0.05
        top_k = 100

        df = generate_population(
            pop_size=pop_size,
            baseline_prevalence=baseline_prevalence,
            top_k=top_k,
        )

        # Check basic structure
        self.assertEqual(len(df), pop_size)
        self.assertIn("patient_id", df.columns)
        self.assertIn("true_risk", df.columns)
        self.assertIn("pred_score", df.columns)

        # Check risk bounds
        self.assertTrue(all(0 <= r <= 1 for r in df["true_risk"]))
        self.assertTrue(all(0 <= r <= 1 for r in df["pred_score"]))

    @patch("builtins.print")
    def test_calibrate_model_score(self, mock_print):
        # Create a copy of the test dataframe
        df = self.df.copy()

        # Test calibration
        top_k = 100
        target_ppv = 0.1
        calibrate_model_score(df, top_k, target_ppv, initial_noise=0.05, max_attempts=3)

        # Check that pred_score is now available
        self.assertIn("pred_score", df.columns)

        # Check that all scores are in valid probability range
        self.assertTrue(all(0 <= r <= 1 for r in df["pred_score"]))

    @patch("builtins.print")
    def test_simulate_monthly_intervention(self, mock_print):
        # Test monthly intervention simulation
        df = self.df.copy()
        top_k = 50
        months = 3
        intervention_efficacy = 0.3

        result_df, thresholds = simulate_monthly_intervention(
            df,
            intervention_efficacy=intervention_efficacy,
            top_k=top_k,
            months=months,
        )

        # Check that new columns were added
        self.assertIn("treated", result_df.columns)
        self.assertIn("treatment_month", result_df.columns)
        self.assertIn("final_outcome", result_df.columns)

        # Check that thresholds were recorded
        self.assertEqual(len(thresholds), months)
        self.assertTrue(all("threshold" in t for t in thresholds))

        # Check that some patients were treated
        self.assertTrue(result_df["treated"].sum() > 0)

        # Check that treatment months are as expected
        treated_months = result_df.loc[result_df["treated"], "treatment_month"].unique()
        self.assertTrue(all(m in treated_months for m in range(months)))

        # Check outcomes are binary
        self.assertTrue(set(result_df["final_outcome"].unique()).issubset({0, 1}))

    @patch("builtins.print")
    def test_run_rd_analysis(self, mock_print):
        # Prepare data for RD analysis
        df = self.df.copy()
        df["treated"] = False
        df["treatment_month"] = np.nan
        df["final_outcome"] = 0

        # Simulate setting threshold at 80th percentile of pred_score
        cutoff = df["pred_score"].quantile(0.8)
        df.loc[df["pred_score"] >= cutoff, "treated"] = True
        df.loc[df["pred_score"] >= cutoff, "treatment_month"] = 0

        # Simulate outcomes: treated patients have 30% less risk
        df.loc[df["treated"], "final_outcome"] = np.random.binomial(n=1, p=df.loc[df["treated"], "true_risk"] * 0.7)
        df.loc[~df["treated"], "final_outcome"] = np.random.binomial(n=1, p=df.loc[~df["treated"], "true_risk"])

        # Thresholds list
        thresholds = [
            {
                "month": 0,
                "threshold": cutoff,
                "n_selected": df["treated"].sum(),
            }
        ]

        # Run RD analysis
        sub_df, model, results = run_rd_analysis(df, thresholds, month=0, bandwidth=0.05)

        # Check results
        self.assertIsNotNone(sub_df)
        self.assertIsNotNone(model)
        self.assertIn("simple", results)
        self.assertIn("interaction", results)

        # Check that coefficients and p-values are available
        self.assertIn("coef", results["simple"])
        self.assertIn("pvalue", results["simple"])
        self.assertIn("conf_int", results["simple"])

    @patch("src.claude_run.generate_population")
    @patch("src.claude_run.simulate_monthly_intervention")
    @patch("src.claude_run.run_rd_analysis")
    @patch("src.claude_run.plot_rd_results")
    def test_run_simulation_rd(self, mock_plot, mock_rd, mock_simulate, mock_generate):
        # Create mocks for the functions called in run_simulation_rd
        mock_df = self.df.copy()
        mock_generate.return_value = mock_df

        mock_simulated_df = mock_df.copy()
        mock_simulated_df["treated"] = False
        mock_simulated_df.iloc[:100, mock_simulated_df.columns.get_loc("treated")] = True
        mock_simulated_df["treatment_month"] = np.nan
        mock_simulated_df.loc[mock_simulated_df["treated"], "treatment_month"] = 0
        mock_simulated_df["final_outcome"] = 0
        mock_thresholds = [{"month": 0, "threshold": 0.1, "n_selected": 100}]
        mock_simulate.return_value = (mock_simulated_df, mock_thresholds)

        mock_sub_df = mock_simulated_df.head(200).copy()
        mock_model = MagicMock()
        mock_model.params = {
            "Intercept": 0.05,
            "rd_treat": -0.02,
            "dist": 0.1,
            "rd_treat:dist": 0.05,
        }
        mock_model.bse = {"rd_treat": 0.01}
        mock_model.pvalues = {"rd_treat": 0.04}
        mock_model.conf_int.return_value = pd.DataFrame({0: [-0.04], 1: [0.0]}, index=["rd_treat"])

        mock_results = {
            "simple": {
                "model": mock_model,
                "coef": -0.02,
                "se": 0.01,
                "pvalue": 0.04,
                "conf_int": [-0.04, 0.0],
            },
            "interaction": {
                "model": mock_model,
                "coef": -0.02,
                "se": 0.01,
                "pvalue": 0.04,
                "conf_int": [-0.04, 0.0],
            },
        }

        mock_rd.return_value = (mock_sub_df, mock_model, mock_results)
        mock_plot.return_value = MagicMock()

        # Call the function
        results = run_simulation_rd(
            pop_size=1000,
            baseline_prevalence=0.05,
            top_k_factor=2,
            intervention_efficacy=0.3,
            top_k=100,
            months=3,
            bandwidth=0.02,
            random_seed=42,
        )

        # Check that the function returned results
        self.assertIsNotNone(results)
        self.assertIn("population_df", results)
        self.assertIn("monthly_thresholds", results)
        self.assertIn("rd_results", results)
        self.assertIn("summary", results)


if __name__ == "__main__":
    unittest.main()
