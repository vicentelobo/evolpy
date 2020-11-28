import numpy as np

from time import time
from collections import OrderedDict
from random import random, uniform, choice, randint, sample
from tqdm.auto import tqdm

from evolpy import v_print, num_digits

class GA:

    """
    Class to implement genetic algorithm to minimization.
    """

    def __init__(self, fitness, parameters, populationSize, maxGen,
                 retain=0.2, randomSelect=0.05, mutationRate=0.01, 
                 gradeThreshold=1e-8, stabilizationPatience=10,
                 crossoverMethod='one-point', crossoverRate=.25, 
                 selectionMethod='roulette', replacementMethod='steady_state', 
                 stopCriteria='num_iterations', history={}, verbose=False):
        """
        Create an instance of the :class:`.GA`.


        :param fitness: The function used to get the fitness of a population.
        :param parameters: Ordered dictionary the parameters being minimized. 
                           It must contain the name and limits in the
                           format `name: (lower, upper)`.
        :param populationSize: The size of the population which will be evolved by the optimizer.
        :param maxGen: The max number of generations.

        
        :param retain: The percentage of elements retained in the truncation. Must be between 0 and 1.
        :param randomSelect: The chance of random selection of an individual. Must be between 0 and 1.
        :param mutationRate: The chance of mutating an individual. Must be between 0 and 1.
        :param gradeThreshold: The threshold to consider the fitness as stabilized.
        :param stabilizationPatience: The number of iterations that must have the same fitness 
                                      value before stopping the evolution.
        :param crossoverMethod: The crossover method. Options: one-point or two-point.
        :param crossoverRate: The chance of doing the crossover. Must be between 0 and 1.
        :param selectionMethod: The selection method. Options: random, tournament or roulette.
        :param replacementMethod: The replacement method. Options: steady_state, elitism or exchange
        :param stopCriteria: The stop criteria. Options: num_iterations or fitness_stabilized.
        :param verbose: Show progress bars.
        """
        assert isinstance(parameters, OrderedDict),\
               "To unexpected behavior, the parameters must be of type OrderedDict"

        self.fitness = fitness
        self.parameters = parameters
        self.populationSize = populationSize
        self.maxGen = maxGen
        self.retain = retain
        self.randomSelect = randomSelect
        self.mutationRate = mutationRate
        self.history = history
        self.stop = False
        self.gradeThreshold = gradeThreshold
        self.crossoverMethod = crossoverMethod
        self.crossoverRate = crossoverRate
        self.selectionMethod = selectionMethod
        self.stopCriteria = stopCriteria
        self.stabilizationPatience = stabilizationPatience
        self.replacementMethod = replacementMethod
        self.verbose = verbose
        self.v_print = v_print if verbose else lambda *a, **k: None

    def individual(self):
        """
        Create a member of the population.
        """
        return [round(uniform(*parameter), num_digits) 
                if type(parameter) == tuple else choice(parameter)
                for parameter in self.parameters.values()]


    def population(self, count):
        """
        Instantiate a population with `count` individuals.

        :param count: The number of individuals that must be in the population.
        """ 
        pop = []
        while len(pop) < count:
            ind = self.individual()
            if ind not in pop:
                pop.append(ind)
        return pop
 

    def grade(self, list_fit=None):
        """
        Find minimum fitness for population.

        :param list_fit: The array of fitness to be graded.
        """
        if not list_fit:
            list_fit = self.fit
        try:
            return np.nanmin([fit for fit in self.fit])
        except:
            return np.nan

    def evaluate_individual(self, individual):
        """
        Evaluate the fitness of a single individual.

        :param individual: The individual being evaluated.
        """
        individual_tuple = tuple(individual)
        if individual_tuple not in self.history:
            fitness = self.fitness(individual)
            if np.isnan(fitness):
                fitness = self.fitness(individual)
            self.history[individual_tuple] = fitness
        return self.history[individual_tuple]

    def evaluate_population(self, population):
        """
        Evaluate the fitness of the entire population.

        :param population: The population being evaluated.
        """
        return [self.evaluate_individual(individual)
                for individual in tqdm(population,
                                       desc="    Evaluating",
                                       disable=not(self.verbose))]
    
    def replacement(self):
        """
        Determines the rate of individuals to be kept from the
        current generation to the next based on the hyperparameter
        `replacementMethod` set in the evolver instantiation.
        """
        method = self.replacementMethod
        if method == 'elitism':
            retain_length = 1
        elif method == 'steady_state' : 
            # Steady State ==> M individuals from the current generation 
            # are maintained to the next one
            retain_length = int(self.populationSize*self.retain)
        else:
            # method= 'exchange': #Exchange the entire population
            retain_length = 0
        return retain_length


    def crossover(self, solution1, solution2):
        """
        Random One-Point crossover or Specific Crossover Point

        :param solution1: The first individual that will be used to the crossover.
        :param solution2: The second individual that will be used to the crossover.
        """
        if np.random.uniform(0,1) < self.crossoverRate:       
            method = self.crossoverMethod
            interval = len(solution1)
           
            if method == 'two-point':
                crossover_point1, crossover_point2 = sorted(sample(range(interval), k=2))
                child1 =  solution1[:crossover_point1] + solution2[crossover_point1:crossover_point2] + solution1[crossover_point2:]
                child2 = solution2[:crossover_point1] + solution1[crossover_point1:crossover_point2] + solution2[crossover_point2:]

            else:
                # method == 'one-point':
                crossover_point = np.random.randint(interval)
                child1 = solution1[:crossover_point] + solution2[crossover_point:]
                child2 = solution2[:crossover_point] + solution1[crossover_point:]
           
            return child1, child2
        
        else:
            return solution1, solution2

    
    def mutation(self, solution):
        """
        Random mutation of individuals.

        :param solution: The individual who might mutate.
        """
        if np.random.uniform(0,1) < self.mutationRate:
            selected_gene = np.random.randint(len(solution))
            parameter = list(self.parameters.values())[selected_gene]
            solution[selected_gene] = uniform(*parameter) if type(parameter) == tuple else choice(parameter)

    def selection(self, k=2):
        """
        The method to select new individuals to the crossover.
        Implemented methods are: Tournament Selection, Roulette or Random.
        Don't forget that this implementation aims to minimize!

        :param k: The number of individuals to be selected.
        """
        method  = self.selectionMethod

        if method == 'roulette':               

            fitness = [1/fit_ for fit_ in self.fit]
            total_fit = float(sum(fitness))
            relative_fitness = [f/total_fit for f in fitness]
            probabilities = [sum(relative_fitness[:i+1]) for i in range(len(relative_fitness))]
            probabilities[-1] = 1.0

            chosen = []
            for n in range(k):
                probability = np.random.uniform(0,1)
                for (i, individual) in enumerate(self.pop):
                    if probability <= probabilities[i]:
                        chosen.append(individual)
                        break
            return chosen
        
        
        if method == 'tournament':
            bests = []
            while len(bests) < k:
                # cand1, cand2 = sample(self.pop, k)
                    # Pick individuals for tournament
                fighter_1 = np.random.randint(len(self.pop))
                fighter_2 = np.random.randint(len(self.pop))

                    # Get fitness score for each
                fighter_1_fitness = self.fit[fighter_1]
                fighter_2_fitness = self.fit[fighter_2]
                                        
                if fighter_1_fitness <= fighter_2_fitness:
                    winner = self.pop[fighter_1]
                else:
                    winner = self.pop[fighter_2]
                
                bests.append(winner)
            return bests

        else:
            """
              Random selection
            """
            return sample(self.pop, k)



    def evolve(self):
        """
        Method to evolve our population.
        - First, we apply elitism to guarantee that the fittest element will be in the next generation 
        - Then, we add randomized individuals to promote genetic diversity;
        - Then we select random individuals to be mutated;
        - Finally, we apply crossover if the population did not converge.
        """
        # ------------------------------------------------------------------
        # Replace Population
        # ------------------------------------------------------------------
        
        # retain_length = int(self.populationSize*self.retain)
        retain_length = self.replacement()
        parents = [ind for ind in tqdm(self.pop[:retain_length], 
                                       desc="    Retain",
                                       disable=not(self.verbose))]

        # ------------------------------------------------------------------
        # RANDOM SELECTION --> Add random individuals to promote diversity
        # ------------------------------------------------------------------
        
        if len(parents) < self.populationSize:
          for individual in tqdm(self.pop[retain_length:], 
                                 desc="    Random",
                                 disable=not(self.verbose)):
              if self.randomSelect > random():
                  parents.append(individual)

        # ------------------------------------------------------------------
        # Mutate random individuals
        # ------------------------------------------------------------------
        
        for individual in tqdm(parents, 
                               desc="    Mutation",
                               disable=not(self.verbose)):
            self.mutation(individual)
        
        # ------------------------------------------------------------------
        # Crossover individuals if the population has not converged
        # ------------------------------------------------------------------
        
        desired_length = self.populationSize - len(parents)
        with tqdm(total=desired_length, 
                  desc="    Mutation",
                  disable=not(self.verbose)) as pbar:
            while len(parents) < self.populationSize:
                unique = np.unique(self.pop, axis=0)
                
                if(len(unique) < 2):
                    self.v_print("  # of different elements < 2")
                    extend = [self.pop[0]] * desired_length
                    parents.extend(extend)
                    pbar.update(desired_length)
                else:
                    # SELECTION
                    solution1, solution2 = self.selection(k=2)

                    cont = 0
                    while solution1 == solution2:
                        cont += 1
                        solution2 = self.selection(k=1)[0]
                        if cont > 100:
                            solution2 = self.individual()
                    
                    
                    # CROSSOVER
                    child1, child2 = self.crossover(solution1, solution2)
                    parents.append(child1)
                    parents.append(child2)
                    pbar.update(2)

        # ------------------------------------------------------------------
        # Find best individuals
        # ------------------------------------------------------------------

        # Sort the new population before printing
        parents = sorted(parents[:self.populationSize])
        self.v_print("    New Population:", parents)

        # Evaluate the fitness of every individual
        new_fit = self.evaluate_population(parents)

        # Find the best fitness in the population
        new_best = self.grade(new_fit)
        self.v_print("    Best fitness of this generation:", new_best)

        # Sort the new population by fitness
        self.fit, self.pop = [list(t) for t in zip(*sorted(zip(new_fit, parents)))]
        self.best_fit = new_best

        return self.best_fit

    def report(self, grade_history):
        """
        Return the best gene and its corresponding fitness

        :param grade_history: The array containing all grades obtained during the evolution.
        """

        index = np.argmin(self.fit)
        best = {
            'gene': dict(zip(self.parameters.keys(), self.pop[index])),
            'fitness': self.fit[index]
        }
        return best, grade_history
      
    def run(self):
        """
        Run the genetic algorithm until we reach the maximum number of generations.
        """
        self.pop = sorted(self.population(self.populationSize))
        self.v_print("    Population:", self.pop)
        
        self.fit = self.evaluate_population(self.pop)
        self.best_fit = self.grade()
        self.v_print("    Initial best:", self.best_fit)
        
        self.fit, self.pop = [list(t) for t in zip(*sorted(zip(self.fit, self.pop)))]
        grade_history = []
        
        old_grade =  0
        patience = self.stabilizationPatience
                
        # Run for `maxGen` generations
        for i in tqdm(range(self.maxGen), desc="Generations"):
            t = time()
            self.v_print("Running generation {}/{}".format(i+1, self.maxGen))
            grade = self.evolve()
            grade_history.append(grade)
            
            if self.stopCriteria == 'fitness_stabilized':
                if np.abs(grade - old_grade) <= self.gradeThreshold:
                    if patiency > 0:
                        patiency -= 1
                    else:
                        self.v_print("  Old grade: .", old_grade)
                        self.v_print("  Current grade: .", grade)
                        self.v_print("  Ending execution: fitness stabilized.")
                        break
                else:
                    patiency = self.stabilizationPatience
            
            old_grade = grade
            
            self.v_print("  End of generation.\n  Time elapsed (s):", time() - t)
        
        return self.report(grade_history)