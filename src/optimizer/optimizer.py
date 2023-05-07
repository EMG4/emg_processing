#!/usr/bin/env python3
#==============================================================================
# Author: Carl Larsson
# Description: Optimization of classifier parameters
# Date: 2023-05-03
#==============================================================================


import pygad.kerasga
import numpy
import tensorflow.keras

import config
import math
import pprint
import sys


#==============================================================================
# Fitness function for calculating the fitness of a solution
def fitness_func(ga_instance, solution, sol_idx):

    numpy.set_printoptions(threshold=sys.maxsize)

    # Obtain the parameters from one solution
    model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=config.model, weights_vector=solution)
    
    # Use the solutions parameters to set the models parameters
    config.model.set_weights(weights=model_weights_matrix)

    # The model predicts on all the data
    predictions = config.model.predict(config.ga_data, verbose = 0)

    # We check how well it predicted and base it's fitness on how good it predicted
    cce = tensorflow.keras.losses.CategoricalCrossentropy()
    # Add the small value so we don't divide by 0
    solution_fitness = 1.0 / (cce(config.ga_labels, predictions).numpy() + 0.00000001)

    if(math.isnan(solution_fitness)):
        print("dad had a good idea")
        print(cce(config.ga_labels, predictions).numpy())
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(config.ga_labels)
        pp.pprint(predictions)
        solution_fitness = 0
        config.nan_times = config.nan_times + 1
        print(config.nan_times)

    # Return the fitness
    return solution_fitness
#==============================================================================
# Used for debugging, tells the progress of the GA
def callback_generation(ga_instance):
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
#==============================================================================
# Genetic Algorithm for optimization of classifier parameters
def ga(num_solutions, num_generations, num_parents_mating):

    # Which model to optimize and the number of soultions in the population
    keras_ga = pygad.kerasga.KerasGA(model=config.model, num_solutions=num_solutions)

    # Cannot have more parents than solutions in the population
    if(num_parents_mating > num_solutions):
        print(f"Number of parents mating was set to", {num_solutions}, "(down from", {num_parents_mating}, "), because number of parents mating must be less than number of soultions in the population")
        num_parents_mating = num_solutions

    # Create the initial population
    initial_population = keras_ga.population_weights

    config.nan_times = 0
    # Number of generations and parents mating in the GA
    # Initial population of the GA
    # Which fitness function the GA should use
    ga_instance = pygad.GA(num_generations=num_generations, num_parents_mating=num_parents_mating, initial_population=initial_population, fitness_func=fitness_func, on_generation=callback_generation)
    
    # Run the GA
    ga_instance.run()

    # Get best solution and parameters
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    best_solution_weights = pygad.kerasga.model_weights_as_matrix(model=config.model, weights_vector=solution)

    # Return the best parameters found
    return best_solution_weights
#==============================================================================
