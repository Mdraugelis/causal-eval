import unittest
import numpy as np
from src.intervention_sim import RiskModel, Person, StrokeSimulationWithIntervention

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
        self.assertFalse(person.intervention_applied)
        self.assertIsNone(person.intervention_month)

class TestStrokeSimulationWithIntervention(unittest.TestCase):
    def setUp(self):
        self.sim = StrokeSimulationWithIntervention(
            population_size=100,
            annual_incidence_rate=0.05,
            num_years=1,
            seed=42,
            intervention_effectiveness=0.3,
            num_interventions=10
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
        
        # Check intervention parameters
        self.assertEqual(self.sim.intervention_effectiveness, 0.3)
        self.assertEqual(self.sim.num_interventions, 10)
        
        # Check that everyone is initially stroke-free, alive, and without intervention
        for person in self.sim.population:
            self.assertFalse(person.has_stroke)
            self.assertTrue(person.is_alive)
            self.assertFalse(person.intervention_applied)
            self.assertIsNone(person.intervention_month)
    
    def test_age_distribution(self):
        # Create a simulation with age distribution
        sim_with_age = StrokeSimulationWithIntervention(
            population_size=100,
            annual_incidence_rate=0.05,
            num_years=1,
            seed=42,
            age_distribution=True,
            initial_age_range=(20, 80),
            intervention_effectiveness=0.3,
            num_interventions=10
        )
        
        # Check that ages are assigned and within range
        for person in sim_with_age.population:
            self.assertIsNotNone(person.age)
            self.assertGreaterEqual(person.age, 20)
            self.assertLessEqual(person.age, 80)
    
    def test_intervention_limits(self):
        # Run a partial simulation to check intervention assignment
        for month in range(3):  # Run only 3 months for testing
            # Age update at yearly intervals
            if month > 0 and (month % 12 == 0):
                self.sim._age_update(month)
            
            # Apply mortality if enabled
            if self.sim.include_mortality:
                self.sim._apply_mortality(month)
            
            # Simulate risk scores and intervention assignment
            current_risk_scores = {}
            for person in self.sim.population:
                if person.is_alive and not person.has_stroke and not person.intervention_applied:
                    risk = self.sim.risk_model.predict_risk(is_positive=False)
                    person.monthly_risk_scores.append(risk)
                    current_risk_scores[person.person_id] = risk
                else:
                    person.monthly_risk_scores.append(None)
            
            # Get eligible patients for intervention (not already intervened)
            eligible = [p for p in self.sim.population if p.is_alive and not p.has_stroke and not p.intervention_applied]
            eligible_sorted = sorted(eligible, key=lambda p: current_risk_scores.get(p.person_id, 0), reverse=True)
            num_to_intervene = min(self.sim.num_interventions, len(eligible_sorted))
            intervened_patients = eligible_sorted[:num_to_intervene]
            for p in intervened_patients:
                p.intervention_applied = True
                p.intervention_month = month
        
        # After partial simulation, count interventions
        total_interventions = sum(1 for p in self.sim.population if p.intervention_applied)
        
        # We should have at most 30 interventions (10 per month for 3 months)
        # But might be less if some people had strokes or died
        self.assertLessEqual(total_interventions, 30)
        
        # Check that interventions were applied to highest risk people
        for month in range(3):
            # Get people who received intervention in this month
            intervened_this_month = [p for p in self.sim.population if p.intervention_applied and p.intervention_month == month]
            
            # If nobody was intervened this month, skip
            if not intervened_this_month:
                continue
                
            # Get lowest risk score among intervened patients
            lowest_intervened_score = min(p.monthly_risk_scores[month] for p in intervened_this_month)
            
            # Check that nobody with a higher risk score was left without intervention
            for p in self.sim.population:
                if (not p.intervention_applied and 
                    p.is_alive and 
                    not p.has_stroke and
                    len(p.monthly_risk_scores) > month and
                    p.monthly_risk_scores[month] is not None and
                    p.monthly_risk_scores[month] > lowest_intervened_score):
                    # This should never happen - higher risk people should get intervention first
                    self.fail(f"Person {p.person_id} had higher risk ({p.monthly_risk_scores[month]}) than intervened person ({lowest_intervened_score}) but didn't get intervention")

if __name__ == '__main__':
    unittest.main()