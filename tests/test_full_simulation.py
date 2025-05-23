import unittest
from src.strokesimulation import StrokeSimulation
from src.intervention_sim import StrokeSimulationWithIntervention

class TestFullSimulation(unittest.TestCase):
    def test_stroke_simulation_run(self):
        sim = StrokeSimulation(population_size=20, annual_incidence_rate=0.05, num_years=1, seed=0)
        sim.run_simulation()
        # Ensure counts recorded for each month
        self.assertEqual(len(sim.monthly_stroke_counts), sim.total_months)
        self.assertEqual(len(sim.monthly_alive_counts), sim.total_months)

    def test_intervention_simulation_run(self):
        sim = StrokeSimulationWithIntervention(population_size=20, annual_incidence_rate=0.05, num_years=1, seed=0, intervention_effectiveness=0.2, num_interventions=5)
        sim.run_simulation()
        # After full run, we should have logs for each month
        self.assertEqual(len(sim.intervention_log), sim.total_months)
        self.assertEqual(len(sim.monthly_stroke_counts), sim.total_months)
        self.assertEqual(len(sim.monthly_alive_counts), sim.total_months)

if __name__ == '__main__':
    unittest.main()
