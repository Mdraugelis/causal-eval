import unittest
import numpy as np
from src.strokesimulation import RiskModel, Person, StrokeSimulation

class TestRiskModel(unittest.TestCase):
    def setUp(self):
        self.risk_model = RiskModel()
        np.random.seed(42)  # For reproducibility
    
    def test_predict_risk_positive(self):
        # Test positive prediction
        risk = self.risk_model.predict_risk(is_positive=True)
        self.assertGreaterEqual(risk, 0)
        self.assertLessEqual(risk, 1)
    
    def test_predict_risk_negative(self):
        # Test negative prediction
        risk = self.risk_model.predict_risk(is_positive=False)
        self.assertGreaterEqual(risk, 0)
        self.assertLessEqual(risk, 1)
    
    def test_invalid_parameters(self):
        # Test validation for right-skewed distribution
        with self.assertRaises(ValueError):
            self.risk_model.predict_risk(
                is_positive=False, 
                neg_alpha=1.0,  # Should be < 1
                neg_beta=1.1
            )
        
        # Test validation for left-skewed distribution
        with self.assertRaises(ValueError):
            self.risk_model.predict_risk(
                is_positive=True,
                pos_alpha=1.0,  # Should be > 1
                pos_beta=1.0    # Should be < 1
            )
    
    def test_invalid_is_positive(self):
        # Test validation for is_positive parameter
        with self.assertRaises(ValueError):
            self.risk_model.predict_risk(is_positive=None)

class TestPerson(unittest.TestCase):
    def test_person_initialization(self):
        person = Person(person_id=1)
        
        self.assertEqual(person.person_id, 1)
        self.assertFalse(person.has_stroke)
        self.assertIsNone(person.stroke_month)
        self.assertTrue(person.is_alive)
        self.assertIsNone(person.age)
        self.assertEqual(person.risk_factors, {})
        self.assertEqual(person.monthly_risk_scores, [])

class TestStrokeSimulation(unittest.TestCase):
    def setUp(self):
        self.sim = StrokeSimulation(
            population_size=100,
            annual_incidence_rate=0.05,
            num_years=1,
            seed=42
        )
    
    def test_initialization(self):
        # Check population size
        self.assertEqual(len(self.sim.population), 100)
        
        # Check monthly probability calculation
        expected_monthly_prob = 1 - (1 - 0.05) ** (1/12)
        self.assertAlmostEqual(self.sim.monthly_stroke_prob, expected_monthly_prob)
        
        # Check initialization of counters
        self.assertEqual(len(self.sim.monthly_stroke_counts), 12)  # 1 year = 12 months
        self.assertEqual(len(self.sim.monthly_alive_counts), 12)
        
        # Check that everyone is initially stroke-free and alive
        for person in self.sim.population:
            self.assertFalse(person.has_stroke)
            self.assertTrue(person.is_alive)
    
    def test_age_distribution(self):
        # Create a simulation with age distribution
        sim_with_age = StrokeSimulation(
            population_size=100,
            annual_incidence_rate=0.05,
            num_years=1,
            seed=42,
            age_distribution=True,
            initial_age_range=(20, 80)
        )
        
        # Check that ages are assigned and within range
        for person in sim_with_age.population:
            self.assertIsNotNone(person.age)
            self.assertGreaterEqual(person.age, 20)
            self.assertLessEqual(person.age, 80)

if __name__ == '__main__':
    unittest.main()