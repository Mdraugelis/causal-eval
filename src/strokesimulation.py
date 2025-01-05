import random
import math
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Person:
    """Class to hold information about each individual in the simulation."""
    person_id: int
    has_stroke: bool = False
    stroke_month: Optional[int] = None
    is_alive: bool = True
    
    # Optional: you can include extra attributes for age, risk factors, etc.
    age: Optional[int] = None
    risk_factors: dict = field(default_factory=dict)

class StrokeSimulation:
    """
    A Python-based simulation of stroke incidence in a population, 
    following the provided design guide.
    """
    
    def __init__(self,
                 population_size: int,
                 annual_incidence_rate: float,
                 num_years: int,
                 seed: Optional[int] = None,
                 include_mortality: bool = False,
                 annual_mortality_rate: float = 0.01,
                 age_distribution: bool = False,
                 initial_age_range: tuple = (20, 80)):
        """
        Initialize the simulation with key parameters.
        
        :param population_size: Number of individuals to simulate.
        :param annual_incidence_rate: Base probability of stroke per year (e.g., 0.002 for 0.2%).
        :param num_years: Number of years to run the simulation.
        :param seed: Random seed for reproducibility (optional).
        :param include_mortality: Whether to model deaths from non-stroke causes.
        :param annual_mortality_rate: Probability of dying (non-stroke) per year if include_mortality is True.
        :param age_distribution: Whether to assign each person an initial age.
        :param initial_age_range: Tuple specifying min and max possible starting age (used if age_distribution=True).
        """
        
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
        # Approximate transformation from annual to monthly incidence:
        # p_monthly = 1 - (1 - p_annual)**(1/12)
        self.monthly_stroke_prob = 1 - (1 - self.annual_incidence_rate) ** (1/12)
        self.monthly_mortality_prob = None
        if include_mortality:
            self.monthly_mortality_prob = 1 - (1 - self.annual_mortality_rate) ** (1/12)
        
        # List of Person objects
        self.population: List[Person] = []
        
        # For output logging
        self.stroke_log = []  # will store tuples of (person_id, month_of_stroke)
        self.monthly_stroke_counts = [0] * self.total_months
        self.monthly_alive_counts = [0] * self.total_months
        
        # Initialize population
        self._initialize_population()
    
    def _initialize_population(self):
        """Create and store Person objects for the simulation."""
        for i in range(self.population_size):
            # Optional: assign random age if requested
            age = None
            if self.age_distribution:
                age = random.randint(self.initial_age_range[0],
                                     self.initial_age_range[1])
            
            person = Person(
                person_id=i,
                has_stroke=False,
                stroke_month=None,
                is_alive=True,
                age=age
            )
            self.population.append(person)
    
    def run_simulation(self):
        """
        Run the month-by-month simulation.
        For each month:
         - Optionally update ages.
         - Optionally check mortality (non-stroke).
         - Check stroke occurrence.
         - Track how many are still alive and stroke-free.
        """
        
        for month in range(self.total_months):
            # (Optional) Age each person by 1/12th of a year
            self._age_update(month)
            
            # Mortality check (if enabled)
            if self.include_mortality:
                self._apply_mortality(month)
            
            # Stroke check
            self._apply_stroke(month)
            
            # Count how many are alive (for optional monthly tracking)
            alive_count = sum(p.is_alive for p in self.population)
            self.monthly_alive_counts[month] = alive_count
    
    def _age_update(self, month: int):
        """Increment age by 1 year each 12-month interval (if age distribution is used)."""
        if self.age_distribution and month > 0 and (month % 12 == 0):
            for person in self.population:
                if person.is_alive:
                    person.age += 1
    
    def _apply_mortality(self, month: int):
        """
        For each alive individual, draw a random number to determine if they die
        from non-stroke causes this month.
        """
        for person in self.population:
            if person.is_alive and not person.has_stroke:
                if random.random() < self.monthly_mortality_prob:
                    person.is_alive = False
    
    def _apply_stroke(self, month: int):
        """
        For each alive, stroke-free individual, decide if they have a stroke this month.
        Record the event and store logs accordingly.
        """
        stroke_count_this_month = 0
        
        for person in self.population:
            if person.is_alive and not person.has_stroke:
                # Adjust stroke probability if needed (e.g., by age, risk factors).
                stroke_prob = self.monthly_stroke_prob
                
                # Example of how you might account for age or risk factor:
                # if person.age and person.age > 60:
                #     stroke_prob *= 2  # arbitrary adjustment for demonstration
                
                if random.random() < stroke_prob:
                    person.has_stroke = True
                    person.stroke_month = month
                    self.stroke_log.append((person.person_id, month))
                    stroke_count_this_month += 1
        
        self.monthly_stroke_counts[month] = stroke_count_this_month
    
    def summarize_results(self):
        """
        Provide a final report of key statistics.
        """
        # How many had strokes
        total_strokes = len([p for p in self.population if p.has_stroke])
        # Final stroke rate (as a fraction of initial population)
        stroke_rate_fraction = total_strokes / self.population_size
        stroke_rate_percent = stroke_rate_fraction * 100
        
        # How many are still alive
        final_alive_count = sum(p.is_alive for p in self.population)
        
        # If mortality is modeled, how many died without stroke
        non_stroke_deaths = 0
        if self.include_mortality:
            non_stroke_deaths = len([p for p in self.population 
                                     if (not p.is_alive) and (not p.has_stroke)])
        
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
        
        # Optional: more sophisticated summaries
        # E.g. average month of stroke among those who had one
        stroke_months = [p.stroke_month for p in self.population if p.has_stroke]
        if stroke_months:
            avg_stroke_month = sum(stroke_months) / len(stroke_months)
            print(f"Average Month of First Stroke: {avg_stroke_month:.2f}")
        
        print("\n--- Individual-level Log (first 10 events) ---")
        for event in self.stroke_log[:10]:
            print(f"Person {event[0]} had a stroke in month {event[1]}")
        print("...")
    
    def get_results(self):
        """
        Return structured results for programmatic use:
        - stroke_log: list of (person_id, month_of_stroke)
        - monthly_stroke_counts
        - monthly_alive_counts
        - final population state
        """
        results = {
            "stroke_log": self.stroke_log,
            "monthly_stroke_counts": self.monthly_stroke_counts,
            "monthly_alive_counts": self.monthly_alive_counts,
            "final_population": self.population
        }
        return results

def main():
    # Example usage:
    
    # Configuration (example values):
    N = 600000                  # population size
    annual_incidence = 0.002  # 0.2% annual stroke incidence
    years = 50                # simulate for 10 years
    seed_value = 42           # for reproducibility
    
    # Create the simulation instance
    sim = StrokeSimulation(
        population_size=N,
        annual_incidence_rate=annual_incidence,
        num_years=years,
        seed=seed_value,
        include_mortality=True,
        annual_mortality_rate=0.01,
        age_distribution=True,
        initial_age_range=(30, 70)
    )
    
    # Run the simulation
    sim.run_simulation()
    
    # Print a summary
    sim.summarize_results()
    
    # Retrieve detailed results if needed
    results = sim.get_results()
    # e.g., you could save these to a file or do further analysis

if __name__ == "__main__":
    main()