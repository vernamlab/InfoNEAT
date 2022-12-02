"""Implements the core evolution algorithm."""
from __future__ import print_function
import random
import neat
import sys
from neat.reporting import ReporterSet
from neat.graphs import feed_forward_layers
from neat.math_util import mean
from neat.six_util import iteritems, itervalues
import numpy as np
import itertools
import collections
import os
from operator import itemgetter
from math import log
import algo1_AE


class CompleteExtinctionException(Exception):
    pass


# def batch_data(x, y, m): ### added by Rabin
#     sss = StratifiedShuffleSplit(n_splits=1, test_size = m)
#     x_batch = list()
#     y_batch = list()
#     for i, j in sss.split(np.zeros(len(y)), y):
#         for k in j:
#             x_batch.append(x[k])
#             y_batch.append(y[k])
#     return x_batch, y_batch
# index_label = list()
# for a in y_batch:
#     index_label.append(a.index(max(a)))
# print(Counter(index_label))

def batch_data_ovr(x, y, m): ### added by Rabin
    x_batch = list()
    y_batch = list()
    # x, y = shuffle(x, y)
    if type(x) is not list and type(y) is not list:
        x = x.tolist()
        y = y.tolist()
    i, j = 0, 0

    for a, b in zip(x, y):
        if sum(b) == 1 and i < m/2:
            x_batch.append(a)
            y_batch.append(b)
            i += 1
        elif j < m/2:
            x_batch.append(a)
            y_batch.append(b)
            j += 1
        if len(x_batch) == m:
            break
    return x_batch, y_batch


class Population(object):
    """
    This class implements the core evolution algorithm:
        1. Evaluate fitness of all genomes.
        2. Check to see if the termination criterion is satisfied; exit if it is.
        3. Generate the next generation from the current population.
        4. Partition the new generation into species based on genetic similarity.
        5. Go to 1.
    """

    def __init__(self, config, initial_state=None):
        self.reporters = ReporterSet()
        self.config = config
        stagnation = config.stagnation_type(config.stagnation_config, self.reporters)
        self.reproduction = config.reproduction_type(config.reproduction_config,
                                                     self.reporters,
                                                     stagnation)
        if config.fitness_criterion == 'max':
            self.fitness_criterion = max
        elif config.fitness_criterion == 'min':
            self.fitness_criterion = min
        elif config.fitness_criterion == 'mean':
            self.fitness_criterion = mean
        elif not config.no_fitness_termination:
            raise RuntimeError(
                "Unexpected fitness_criterion: {0!r}".format(config.fitness_criterion))

        if initial_state is None:
            # Create a population from scratch, then partition into species.
            self.population = self.reproduction.create_new(config.genome_type,
                                                           config.genome_config,
                                                           config.pop_size)
            self.species = config.species_set_type(config.species_set_config, self.reporters)
            self.generation = 0
            self.species.speciate(config, self.population, self.generation)
        else:
            self.population, self.species, self.generation = initial_state

        self.stop_evol1, self.stop_evol2 = 0,0
        self.best_genome = None
        self.previous_best_genome_i = None
        self.previous_best_genome_i_minus_1 = None

    def add_reporter(self, reporter):
        self.reporters.add(reporter)

    def remove_reporter(self, reporter):
        self.reporters.remove(reporter)

    def CMI(self, MNIST_inputs, genome1, genome2, MNIST_outputs, alpha=1.01):
        config = self.config
        # net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
        net1 = genome1.net
        layers_a = net1.layers
        nn_a = algo1_AE.outputs_nodes(MNIST_inputs, net1, layers_a)

        # net2 = neat.nn.FeedForwardNetwork.create(genome2, config)
        net2 = genome2.net
        layers_b = net2.layers
        nn_b = algo1_AE.outputs_nodes(MNIST_inputs, net2, layers_b)

        mutual_information_CMI = algo1_AE.nnanalysis_CMI(MNIST_outputs, alpha, layers_b, layers_a, nn_b,
                                                         nn_a)  # layer wise CMI values
        return mutual_information_CMI

    #### to compare accuracy of previous generations to the current
    def step_1_stopping_criteria(self, current_genome, prev_genome_i, prev_genome_i_minus_1, input_batch, output_batch):
        CMI_value_current = self.CMI(input_batch, current_genome, prev_genome_i, output_batch, 1.01)
        CMI_value_previous = self.CMI(input_batch, prev_genome_i, prev_genome_i_minus_1, output_batch, 1.01)
        stop = 0
        CMI_value_current_last_layer = CMI_value_current[-1]
        CMI_value_previous_last_layer = CMI_value_previous[-1]

        if current_genome.fitness < prev_genome_i.fitness and \
                abs(CMI_value_current_last_layer) > abs(CMI_value_previous_last_layer):
            best_temp = prev_genome_i
            stop = 1
            return best_temp,stop
            #self.reporters.found_solution(self.config, self.generation, self.best_genome)  ### report the best genome
        else:
            return current_genome,stop

    def step_2_stopping_criteria(self, current_genome, prev_genome_i, prev_genome_i_minus_1,input_batch, output_batch):
        diff_fitness = (current_genome.fitness - prev_genome_i.fitness)
        # best_temp = None
        stop = 0
        if diff_fitness > 0 and abs(diff_fitness) < 0.005:
            best_temp = prev_genome_i
            stop = 1
            # self.reporters.found_solution(self.config, self.generation, self.best_genome)  ### report the best genome
            return best_temp,stop
        else:
            log_loss_values = [v.fitness for k, v in self.population.items()]  ### get all fitness values
            log_loss_values.sort(reverse=True)  ## sort in descending the fitness values of all genomes
            new_best_genomes = [k for k, v in self.population.items() if
                                v.fitness == log_loss_values[1]]  ## choose second best genome instead
            best_genome_new = self.population[new_best_genomes[0]]
            best_temp, stop = self.step_1_stopping_criteria(best_genome_new, prev_genome_i, prev_genome_i_minus_1,input_batch, output_batch)
            return best_temp,stop

    def run(self, fitness_function, n, n_batch, inputs, outputs): # modified to add batch training
        """
        Runs NEAT's genetic algorithm for at most n generations.  If n
        is None, run until solution is found or extinction occurs.

        The user-provided fitness_function must take only two arguments:
            1. The population as a list of (genome id, genome) tuples.
            2. The current configuration object.

        The return value of the fitness function is ignored, but it must assign
        a Python float to the `fitness` member of each genome.

        The fitness function is free to maintain external state, perform
        evaluations in parallel, etc.

        It is assumed that fitness_function does not modify the list of genomes,
        the genomes themselves (apart from updating the fitness member),
        or the configuration object.
        """

        if self.config.no_fitness_termination and (n is None):
            raise RuntimeError("Cannot have no generational limit with no fitness termination")

        k = 0


        while n is None or k < n:
            k += 1

            self.reporters.start_generation(self.generation)

            ##added by Rabin
            if n_batch == 0:
                fitness_function(list(iteritems(self.population)), self.config, inputs, outputs)  ### modified by Rabin

            elif n % n_batch == 0:
                input_batch, output_batch = batch_data_ovr(inputs, outputs, 256)
                # input_batch, output_batch = batch_data(inputs, outputs, 256)
                fitness_function(list(iteritems(self.population)), self.config, input_batch, output_batch)  ### modified by Rabin
            ##############


            best = None
            #for g in itervalues(self.population):
            for g in itervalues(self.population):
                if g.fitness is None:
                    raise RuntimeError("Fitness not assigned to genome {}".format(g.key))

                if best is None or g.fitness > best.fitness:
                    best = g
            self.reporters.post_evaluate(self.config, self.population, self.species, best)

            # Track the best genome ever seen.
            if self.best_genome is None or best.fitness > self.best_genome.fitness:
                self.best_genome = best

            ###########################################################################################################
            ## genome selection based on fitness and CMI (triggered only when there are multiple genomes with best
            ### values)
            id_best_genomes = [k for k,v in self.population.items() if v.fitness == best.fitness]
            n_best_genomes = len(id_best_genomes)
            CMI_values = list()

            if (n_best_genomes) > 1:
                ## calculate CMI starting from last layer until n_best_CMI is 1
                # CMI(config, MNIST_inputs, genome1, genome2, MNIST_outputs, alpha=1.01) ##genome 1: genomes of this gen,
                                                                                ## genome 2: best genome from prev gen
                for a in id_best_genomes:
                    CMI_values.append(self.CMI(input_batch, best, self.population[a], output_batch,1.01)) ####when n_batch ==0


                try:
                    CMI_values_last_layer = [item[-1] for item in CMI_values]
                except:
                    pass
                count = CMI_values_last_layer.count(min(CMI_values_last_layer)) ###number of genomes with min CMI for the last layer
                if count > 1 and min(CMI_values_last_layer) != 0:
                    CMI_values_last_layer = [item[-2] for item in CMI_values]
                    best_CMI_index = CMI_values_last_layer.index(min(CMI_values_last_layer))
                else:
                    best_CMI_index = CMI_values_last_layer.index(min(CMI_values_last_layer))

                best_genome_CMI_id = id_best_genomes[best_CMI_index]
                self.best_genome = self.population[best_genome_CMI_id]

            #####################################################################################################

            ########### stopping criteria #####################


            stop_evol1, stop_evol2 = 0,0
            if self.previous_best_genome_i is not None and self.previous_best_genome_i_minus_1 is not None and \
                                                                                self.best_genome is not None:
                self.best_genome, self.stop_evol1 = self.step_1_stopping_criteria(self.best_genome, self.previous_best_genome_i, self.previous_best_genome_i_minus_1, input_batch, output_batch)
                self.best_genome, self.stop_evol2 = self.step_2_stopping_criteria(self.best_genome, self.previous_best_genome_i, self.previous_best_genome_i_minus_1, input_batch, output_batch)
            if self.stop_evol1 == 1 or self.stop_evol2 == 1:
                break

            ### after the computation of the stopping criteria
            self.previous_best_genome_i_minus_1 = self.previous_best_genome_i
            self.previous_best_genome_i = self.best_genome
            #####################################################################################################

            if not self.config.no_fitness_termination:
                # End if the fitness threshold is reached.
                fv = self.fitness_criterion(g.fitness for g in itervalues(self.population))
                if fv >= self.config.fitness_threshold:
                    self.reporters.found_solution(self.config, self.generation, best)
                    break

            # Create the next generation from the current generation.
            self.population = self.reproduction.reproduce(self.config, self.species,
                                                          self.config.pop_size, self.generation)

            # Check for complete extinction.
            if not self.species.species:
                self.reporters.complete_extinction()

                # If requested by the user, create a completely new population,
                # otherwise raise an exception.
                if self.config.reset_on_extinction:
                    self.population = self.reproduction.create_new(self.config.genome_type,
                                                                   self.config.genome_config,
                                                                   self.config.pop_size)
                else:
                    raise CompleteExtinctionException()

            # Divide the new population into species.
            self.species.speciate(self.config, self.population, self.generation)

            self.reporters.end_generation(self.config, self.population, self.species)

            self.generation += 1

        if self.config.no_fitness_termination:
            self.reporters.found_solution(self.config, self.generation, self.best_genome)

        return self.best_genome, self.stop_evol1, self.stop_evol2
