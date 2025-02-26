import random
import math, numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
import pandas as pd
import uuid
from datetime import datetime
import logging
from pathlib import Path
import yaml

def load_config(filepath="config.yaml") -> dict:
    with open(filepath, "r") as f:
        config = yaml.safe_load(f)
    return config

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
    """Simulated ML model for stroke risk prediction"""
    def predict_risk(self,
                     is_positive: bool,
                     neg_alpha: float = 0.1,
                     neg_beta: float = 1.1,
                     pos_alpha: float = 1.967,
                     pos_beta: float = 0.7) -> float: 
        """Generate a prediction score using a Beta distribution based on the event type."""
        if is_positive == False:
            if neg_alpha >= 1 or neg_beta <= 1:
                raise ValueError(f"For right-skewed Beta (negative events), expect neg_alpha < 1 and neg_beta > 1; got neg_alpha={neg_alpha}, neg_beta={neg_beta}.")
            return np.random.beta(neg_alpha, neg_beta)
        elif is_positive == True:
            if pos_alpha <= 1 or pos_beta >= 1:
                raise ValueError(f"For left-skewed Beta (positive events), expect pos_alpha > 1 and pos_beta < 1; got pos_alpha={pos_alpha}, pos_beta={pos_beta}.")
            return np.random.beta(pos_alpha, pos_beta)
        else:
            raise ValueError('is_positive must be either False or True.')

@dataclass
class Person:
    """Class to hold information about each individual in the simulation."""
    person_id: int
    has_stroke: bool = False
    stroke_month: Optional[int] = None
    is_alive: bool = True
    age: Optional[int] = None
    risk_factors: dict = field(default_factory=dict)
    monthly_risk_scores: List[Optional[float]] = field(default_factory=list)

class StrokeSimulation:
    """
    A Python-based simulation of stroke incidence in a population.
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
                 risk_model: Optional[RiskModel] = None):
        if seed is not None:
            np.random.seed(seed) 
        
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
        
        self.population: List[Person] = []
        self.stroke_log = []  # Stores tuples of (person_id, month_of_stroke)
        self.monthly_stroke_counts = [0] * self.total_months
        self.monthly_alive_counts = [0] * self.total_months
        self.risk_model = risk_model or RiskModel()
        
        self._initialize_population()

    def run_simulation(self) -> None:
        self.logger.info("Starting population simulation")
        progress_interval = self.total_months // 10

        for month in range(self.total_months):
            if month % progress_interval == 0:
                print(f"Population Simulation {(month/self.total_months)*100:.0f}% complete...")
            self._age_update(month)
            if self.include_mortality:
                self._apply_mortality(month)
            self._apply_stroke(month)
            alive_count = sum(p.is_alive for p in self.population)
            self.monthly_alive_counts[month] = alive_count
        print("Population Simulation complete.") 
        self.logger.info("Population simulation complete")

    def _age_update(self, month: int):
        if self.age_distribution and month > 0 and (month % 12 == 0):
            for person in self.population:
                if person.is_alive:
                    person.age += 1
    
    def _apply_mortality(self, month: int):
        for person in self.population:
            if person.is_alive and not person.has_stroke:
                if random.random() < self.monthly_mortality_prob:
                    person.is_alive = False
    
    def _apply_stroke(self, month: int):
        stroke_count_this_month = 0
        for person in self.population:
            if person.is_alive and not person.has_stroke:
                stroke_prob = self.monthly_stroke_prob
                if random.random() < stroke_prob:
                    person.has_stroke = True
                    person.stroke_month = month
                    self.stroke_log.append((person.person_id, month))
                    stroke_count_this_month += 1
        self.monthly_stroke_counts[month] = stroke_count_this_month

    def run_risk_prediction(self):
        """Original risk prediction method (can be overridden)."""
        progress_interval = self.total_months // 10
        for month in range(self.total_months):
            if month % progress_interval == 0:
                print(f"Risk prediction {(month/self.total_months)*100:.0f}% complete...")
            for person in self.population:
                try:
                    if not person.is_alive:
                        person.monthly_risk_scores.append(None)
                        continue
                    will_have_stroke = False
                    if person.stroke_month is not None:
                        months_until_stroke = person.stroke_month - month
                        if 0 <= months_until_stroke < 12:
                            will_have_stroke = True
                    risk_score = self.risk_model.predict_risk(will_have_stroke)
                    person.monthly_risk_scores.append(risk_score)
                except Exception as e:
                    print(f"Error processing person {person.person_id} at month {month}: {e}")
                    raise
        print("Risk prediction complete.")

    def _initialize_population(self):
        for i in range(self.population_size):
            age = None
            if self.age_distribution:
                age = random.randint(self.initial_age_range[0], self.initial_age_range[1])
            person = Person(
                person_id=i,
                has_stroke=False,
                stroke_month=None,
                is_alive=True,
                age=age
            )
            self.population.append(person)
    
class SimulationRunner(StrokeSimulation):
    """Manages running both population and risk prediction simulations along with monthly analysis"""
    def __init__(self, simulation_params: dict):
        super().__init__(**simulation_params)
        self.simulation_id = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        self.logger = setup_logging(self.simulation_id)
        self.logger.info(f"Initializing simulation {self.simulation_id}")
        self.logger.info(f"Parameters: {simulation_params}")
        self.master_df = pd.DataFrame()      # To accumulate monthly DataFrames
        self.monthly_metrics = []            # To store performance metrics for each month

    def run_all(self) -> dict:
        self.run_simulation()
        self.run_risk_prediction_with_analysis()
        return self.get_results()

    def run_risk_prediction_with_analysis(self):
        """Run risk prediction month by month and perform monthly analysis."""
        progress_interval = self.total_months // 10
        for month in range(self.total_months):
            if month % progress_interval == 0:
                print(f"Risk prediction {(month/self.total_months)*100:.0f}% complete...")
            # Compute risk score for each person for this month
            for person in self.population:
                try:
                    if not person.is_alive:
                        person.monthly_risk_scores.append(None)
                        continue
                    will_have_stroke = False
                    if person.stroke_month is not None:
                        months_until_stroke = person.stroke_month - month
                        if 0 <= months_until_stroke < 12:
                            will_have_stroke = True
                    risk_score = self.risk_model.predict_risk(will_have_stroke)
                    person.monthly_risk_scores.append(risk_score)
                except Exception as e:
                    print(f"Error processing person {person.person_id} at month {month}: {e}")
                    raise
            # After computing risk scores for this month, perform monthly analysis:
            monthly_df = self.analyze_month_data(month)
            # Append this month's DataFrame to the master DataFrame
            if self.master_df.empty:
                self.master_df = monthly_df.copy()
            else:
                self.master_df = pd.concat([self.master_df, monthly_df], ignore_index=True)
        print("Risk prediction and monthly analysis complete.")

    def analyze_month_data(self, month: int) -> pd.DataFrame:
        """Creates a DataFrame for the given month, computes performance metrics,
        and saves the result as a pickle file."""
        monthly_records = []
        for person in self.population:
            risk_score = (person.monthly_risk_scores[month]
                          if len(person.monthly_risk_scores) > month else None)
            had_stroke_within_12 = (person.stroke_month is not None and (0 <= (person.stroke_month - month) < 12))
            record = {
                "Simulation Month": month,
                "Person ID": person.person_id,
                "predicted risk score": risk_score,
                "stroke risk": self.monthly_stroke_prob,  # base risk for the month
                "Had_stroke_within_12_months": had_stroke_within_12,
                "Stroke Month": person.stroke_month,
                "is_alive": person.is_alive,
                "age": person.age
            }
            monthly_records.append(record)
        df_month = pd.DataFrame(monthly_records)
        # Sort by predicted risk score in descending order
        df_month = df_month.sort_values("predicted risk score", ascending=False)
        # Select top 250 patients
        top_250 = df_month.head(250)
        # Compute performance metrics
        true_positives = top_250["Had_stroke_within_12_months"].sum()
        ppv = true_positives / 250
        total_positives = df_month["Had_stroke_within_12_months"].sum()
        sensitivity = true_positives / total_positives if total_positives > 0 else 0
        # Add the metrics as new columns for reference
        df_month["PPV"] = ppv
        df_month["Sensitivity"] = sensitivity
        # Save this month's DataFrame as a pickle file
        df_month.to_pickle(f"month_{month}_analysis.pkl")
        # Save monthly metrics for batch analysis later
        self.monthly_metrics.append({
            "Month": month,
            "PPV": ppv,
            "Sensitivity": sensitivity,
            "True Positives": true_positives,
            "Total Positives": total_positives
        })
        return df_month

    def get_results(self):
        self.logger.info("Saving simulation results")
        # Create results directory
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        simulation_id = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    
        config = {
            'population_size': self.population_size,
            'annual_incidence_rate': self.annual_incidence_rate,
            'num_years': self.num_years,
            'include_mortality': self.include_mortality,
            'annual_mortality_rate': self.annual_mortality_rate,
            'age_distribution': self.age_distribution,
            'initial_age_range': self.initial_age_range,
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
                'simulation_id': simulation_id
            })
        
        df = pd.DataFrame(data)
        # Update file paths
        results_filename = results_dir / f'simulation_results_{simulation_id}.pkl'
        pd.to_pickle({'config': config, 'data': df}, results_filename)
        
        results = {
            "stroke_log": self.stroke_log,
            "monthly_stroke_counts": self.monthly_stroke_counts,
            "monthly_alive_counts": self.monthly_alive_counts,
            "final_population": self.population,
            "dataframe": df,
            "master_dataframe": self.master_df,
            "monthly_metrics": self.monthly_metrics,
            "simulation_id": simulation_id,
            "config": config
        }
        
        self.logger.info(f"Results saved to {results_filename}")
        return results

    def summarize_results(self):
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
        
        print("\n--- Individual-level Log (first 10 events) ---")
        for event in self.stroke_log[:10]:
            print(f"Person {event[0]} had a stroke in month {event[1]}")
        print("...")

def main():
    config = load_config("config.yaml")
    runner = SimulationRunner(config)
    results = runner.run_all()
    runner.summarize_results()
if __name__ == "__main__":
    main()
