This Python script simulates the incidence of strokes in a population over a specified number of years. Here is an overview of its components and functionality:

Person Class:

Represents an individual in the simulation with attributes like person_id, has_stroke, stroke_month, is_alive, age, and risk_factors.
StrokeSimulation Class:

Initializes a simulation with parameters such as population size, annual incidence rate, number of years, random seed, mortality inclusion, annual mortality rate, age distribution, and initial age range.
Converts annual probabilities to monthly probabilities for stroke and mortality.
Initializes a population of Person objects.
Runs the simulation month-by-month, updating ages, checking for mortality and stroke occurrences, and tracking the number of individuals who are still alive and stroke-free.
Provides a summary and detailed results of the simulation.
Main Function:

Configures the simulation parameters and creates an instance of the StrokeSimulation class.
Runs the simulation and prints a summary of the results.
Retrieves detailed results for further analysis if needed.
Key functionalities include:

Simulating stroke events based on a specified annual incidence rate.
Optionally including mortality from non-stroke causes.
Tracking and summarizing the occurrence of strokes and the number of individuals alive at the end of the simulation.
Optionally assigning and updating ages of individuals in the population.
