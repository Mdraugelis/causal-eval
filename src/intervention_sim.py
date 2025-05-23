import random
import math
try:
    import pandas as pd  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pd = None
import uuid
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

def setup_logging(simulation_id: str):
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/simulation_{simulation_id}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

@dataclass
class RiskModel:
    """
    Simulated ML model for stroke risk prediction.
    This implementation uses Beta distributions. In our simulation,
    we use the “negative” (right-skewed) distribution for risk score generation.
    """
    predict_positive_mean: float = 0.8
    predict_positive_std: float = 0.1
    predict_negative_mean: float = 0.2
    predict_negative_std: float = 0.1

    def predict_risk(self,
                     is_positive: bool,
                     neg_alpha: float = 0.1,
                     neg_beta: float = 1.1,
                     pos_alpha: float = 1.967,
                     pos_beta: float = 0.7) -> float:
        if is_positive is False:
            if neg_alpha >= 1 or neg_beta <= 1:
                raise ValueError(f"For right-skewed Beta (negative events), expect neg_alpha < 1 and neg_beta > 1; got neg_alpha={neg_alpha}, neg_beta={neg_beta}.")
            return random.betavariate(neg_alpha, neg_beta)
        elif is_positive is True:
            if pos_alpha <= 1 or pos_beta >= 1:
                raise ValueError(f"For left-skewed Beta (positive events), expect pos_alpha > 1 and pos_beta < 1; got pos_alpha={pos_alpha}, pos_beta={pos_beta}.")
            return random.betavariate(pos_alpha, pos_beta)
        else:
            raise ValueError('is_positive must be either False or True.')

@dataclass
class Person:
    """
    Represents an individual in the simulation.
    Added fields for intervention tracking:
     - intervention_applied: whether the patient has received the intervention
     - intervention_month: the month in which the intervention was applied
    """
    person_id: int
    has_stroke: bool = False
    stroke_month: Optional[int] = None
    is_alive: bool = True
    age: Optional[int] = None
    risk_factors: dict = field(default_factory=dict)
    monthly_risk_scores: List[Optional[float]] = field(default_factory=list)
    intervention_applied: bool = False
    intervention_month: Optional[int] = None

class StrokeSimulationWithIntervention:
    """
    Simulation of a population over time with a monthly intervention.
    
    Each month:
      1. Eligible patients (alive, stroke‑free, and not yet intervened) have a risk score computed.
      2. They are ranked by risk score.
      3. The top N (e.g. 250) receive an intervention that reduces their stroke probability.
      4. Stroke events are simulated using (possibly) modified probabilities.
    """
    def __init__(self,
                 population_size: int,
                 annual_incidence_rate: float,
                 num_years: int,
                 seed: Optional[int] = None,
                 include_mortality: bool = False,
                 annual_mortality_rate: float = 0.01,
                 age_distribution: bool = False,
                 initial_age_range: tuple = (20, 80),
                 risk_model: Optional[RiskModel] = None,
                 intervention_effectiveness: float = 0.3,  # e.g., 30% reduction in stroke risk
                 num_interventions: int = 250):
        if seed is not None:
            random.seed(seed)
        
        self.population_size = population_size
        self.annual_incidence_rate = annual_incidence_rate
        self.num_years = num_years
        self.total_months = num_years * 12
        
        self.include_mortality = include_mortality
        self.annual_mortality_rate = annual_mortality_rate
        
        self.age_distribution = age_distribution
        self.initial_age_range = initial_age_range
        
        # Derived monthly probabilities
        self.monthly_stroke_prob = 1 - (1 - self.annual_incidence_rate) ** (1/12)
        self.monthly_mortality_prob = None
        if include_mortality:
            self.monthly_mortality_prob = 1 - (1 - self.annual_mortality_rate) ** (1/12)
        
        self.risk_model = risk_model or RiskModel()
        self.intervention_effectiveness = intervention_effectiveness
        self.num_interventions = num_interventions
        
        # Logs and results storage
        self.stroke_log = []            # List of tuples: (person_id, month_of_stroke)
        self.monthly_stroke_counts = [0] * self.total_months
        self.monthly_alive_counts = [0] * self.total_months
        # For each month, record the ranking of the top intervention candidates:
        self.intervention_log = []      # List of tuples: (month, [ (person_id, risk_score, intervention_flag), ... ])
        
        self.population: List[Person] = []
        self._initialize_population()

    def _initialize_population(self):
        """Creates the initial population."""
        for i in range(self.population_size):
            age = None
            if self.age_distribution:
                age = random.randint(self.initial_age_range[0], self.initial_age_range[1])
            person = Person(
                person_id=i,
                age=age
            )
            self.population.append(person)
    
    def _age_update(self, month: int):
        """Increase age by 1 year every 12 months (if age_distribution is enabled)."""
        if self.age_distribution and month > 0 and (month % 12 == 0):
            for person in self.population:
                if person.is_alive and person.age is not None:
                    person.age += 1

    def _apply_mortality(self, month: int):
        """Simulate non-stroke deaths for alive individuals."""
        for person in self.population:
            if person.is_alive and not person.has_stroke:
                if random.random() < self.monthly_mortality_prob:
                    person.is_alive = False

    def run_simulation(self):
        """Run the simulation month by month including intervention assignment."""
        print("Starting Simulation with Intervention...")
        for month in range(self.total_months):
            # Age update at yearly intervals
            if month > 0 and (month % 12 == 0):
                self._age_update(month)
            
            # Apply mortality if enabled
            if self.include_mortality:
                self._apply_mortality(month)
            
            # --- Risk Prediction and Intervention Assignment ---
            # For each eligible person (alive, stroke-free, not yet intervened),
            # compute a risk score and record it.
            current_risk_scores = {}
            for person in self.population:
                if person.is_alive and not person.has_stroke and not person.intervention_applied:
                    # Here we use the negative risk prediction (as a proxy)
                    risk = self.risk_model.predict_risk(is_positive=False)
                    person.monthly_risk_scores.append(risk)
                    current_risk_scores[person.person_id] = risk
                else:
                    person.monthly_risk_scores.append(None)
            
            # Get eligible patients for intervention (not already intervened)
            eligible = [p for p in self.population if p.is_alive and not p.has_stroke and not p.intervention_applied]
            # Sort eligible patients by their current month risk score (highest first)
            eligible_sorted = sorted(eligible, key=lambda p: current_risk_scores.get(p.person_id, 0), reverse=True)
            # Determine how many patients to intervene this month
            num_to_intervene = min(self.num_interventions, len(eligible_sorted))
            intervened_patients = eligible_sorted[:num_to_intervene]
            for p in intervened_patients:
                p.intervention_applied = True
                p.intervention_month = month
            
            # Log the ranking for this month (recording only those who got the intervention)
            ranking_info = [(p.person_id, current_risk_scores.get(p.person_id, None), p.intervention_applied) 
                            for p in intervened_patients]
            self.intervention_log.append((month, ranking_info))
            
            # --- Stroke Event Simulation ---
            stroke_count_this_month = 0
            for person in self.population:
                if person.is_alive and not person.has_stroke:
                    # Modify the stroke probability if the patient received the intervention
                    if person.intervention_applied:
                        effective_prob = self.monthly_stroke_prob * (1 - self.intervention_effectiveness)
                    else:
                        effective_prob = self.monthly_stroke_prob
                    if random.random() < effective_prob:
                        person.has_stroke = True
                        person.stroke_month = month
                        self.stroke_log.append((person.person_id, month))
                        stroke_count_this_month += 1
            self.monthly_stroke_counts[month] = stroke_count_this_month
            self.monthly_alive_counts[month] = sum(p.is_alive for p in self.population)
            if month % (self.total_months // 10) == 0:
                print(f"Month {month} complete...")
        print("Simulation with Intervention complete.")

    def get_results(self):
        """Collect simulation results and save to a pickle file."""
        simulation_id = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        config = {
            'population_size': self.population_size,
            'annual_incidence_rate': self.annual_incidence_rate,
            'num_years': self.num_years,
            'include_mortality': self.include_mortality,
            'annual_mortality_rate': self.annual_mortality_rate,
            'age_distribution': self.age_distribution,
            'initial_age_range': self.initial_age_range,
            'intervention_effectiveness': self.intervention_effectiveness,
            'num_interventions': self.num_interventions,
            'simulation_id': simulation_id
        }
        data = []
        for person in self.population:
            data.append({
                'person_id': person.person_id,
                'had_stroke': person.has_stroke,
                'stroke_month': person.stroke_month,
                'is_alive': person.is_alive,
                'age': person.age,
                'risk_scores': person.monthly_risk_scores,
                'intervention_applied': person.intervention_applied,
                'intervention_month': person.intervention_month,
                'simulation_id': simulation_id
            })
        if pd is None:
            raise ImportError("pandas is required for saving results")
        df = pd.DataFrame(data)
        results_filename = f'simulation_results_{simulation_id}.pkl'
        pd.to_pickle({'config': config, 'data': df}, results_filename)
        
        results = {
            "stroke_log": self.stroke_log,
            "monthly_stroke_counts": self.monthly_stroke_counts,
            "monthly_alive_counts": self.monthly_alive_counts,
            "intervention_log": self.intervention_log,
            "final_population": self.population,
            "dataframe": df,
            "simulation_id": simulation_id,
            "config": config
        }
        return results

    def summarize_results(self):
        """Print a summary report of key simulation statistics and intervention details."""
        total_strokes = len([p for p in self.population if p.has_stroke])
        stroke_rate_fraction = total_strokes / self.population_size
        stroke_rate_percent = stroke_rate_fraction * 100
        final_alive_count = sum(p.is_alive for p in self.population)
        non_stroke_deaths = 0
        if self.include_mortality:
            non_stroke_deaths = len([p for p in self.population if (not p.is_alive) and (not p.has_stroke)])
        
        print("=== Simulation Summary ===")
        print(f"Population Size: {self.population_size}")
        print(f"Simulation Duration: {self.num_years} year(s) [{self.total_months} months]")
        print(f"Annual Incidence Rate (input): {self.annual_incidence_rate:.4%}")
        print(f"Derived Monthly Stroke Probability: {self.monthly_stroke_prob:.4%}")
        if self.include_mortality:
            print(f"Annual Mortality Rate (input): {self.annual_mortality_rate:.4%}")
            print(f"Derived Monthly Mortality Probability: {self.monthly_mortality_prob:.4%}")
        print(f"Intervention Effectiveness: {self.intervention_effectiveness:.0%} reduction in stroke risk")
        print(f"Interventions per Month: {self.num_interventions}")
        
        print("\n--- Final Counts ---")
        print(f"Total Strokes: {total_strokes}")
        print(f"Final Stroke Rate: {stroke_rate_percent:.2f}%")
        print(f"Alive at End: {final_alive_count}")
        if self.include_mortality:
            print(f"Non-Stroke Deaths: {non_stroke_deaths}")
        
        print("\n--- Time-Series Info ---")
        print("Monthly stroke counts (first 12 months shown if sim > 12 months):")
        print(self.monthly_stroke_counts[:12], "...")
        
        stroke_months = [p.stroke_month for p in self.population if p.has_stroke]
        if stroke_months:
            avg_stroke_month = sum(stroke_months) / len(stroke_months)
            print(f"Average Month of First Stroke: {avg_stroke_month:.2f}")
        
        print("\n--- Monthly Intervention Log (first 3 months) ---")
        for entry in self.intervention_log[:3]:
            month, ranking = entry
            print(f"Month {month}:")
            for r in ranking:
                pid, risk, applied = r
                print(f"  Person {pid} | Risk Score: {risk:.4f} | Intervention Applied: {applied}")
            print("...")

class SimulationRunnerWithIntervention(StrokeSimulationWithIntervention):
    """Wrapper to manage running the full simulation with intervention and logging."""
    def __init__(self, simulation_params: dict):
        super().__init__(**simulation_params)
        self.simulation_id = f"sim_intervention_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        self.logger = setup_logging(self.simulation_id)
        self.logger.info(f"Initializing simulation {self.simulation_id}")
        self.logger.info(f"Parameters: {simulation_params}")
    
    def run_all(self) -> dict:
        self.run_simulation()
        results = self.get_results()
        return results

    def summarize_results(self):
        super().summarize_results()
        self.logger.info("Simulation summary complete.")

def main():
    simulation_params = {
        "population_size": 40000,
        "annual_incidence_rate": 0.05,  # Input annual stroke incidence rate
        "num_years": 2,
        "seed": 42,
        "include_mortality": True,
        "annual_mortality_rate": 0.01,
        "age_distribution": True,
        "initial_age_range": (30, 70),
        "risk_model": RiskModel(),
        "intervention_effectiveness": 0.2,  # 20% reduction in stroke risk for intervened patients
        "num_interventions": 250
    }
    
    runner = SimulationRunnerWithIntervention(simulation_params)
    results = runner.run_all()
    runner.summarize_results()

if __name__ == "__main__":
    main()
