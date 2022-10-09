from __future__ import print_function
import os
import sys
import neat
import numpy as np
from math import log
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def eval_genome(genome, config, data_in, data_out):
    genome.net = neat.nn.FeedForwardNetwork.create(genome, config)
    sum_score = 0.0
    for xi, xo in zip(data_in, data_out):
        output = genome.net.activate(xi)
        for j in range(len(xo)):
            sum_score += xo[j] * log(1e-15 + output[j])
    mean_sum_score = (1.0 / len(data_out)) * sum_score
    return mean_sum_score, genome.net

def ohe(Y_train, label_class, num_class):
    values = np.array(Y_train)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    Y_profile_ohe = onehot_encoder.fit_transform(integer_encoded)
    classes = list(range(num_class))
    classes.remove(label_class)
    Y_profile_ohe[:, tuple(classes)] = 0
    return Y_profile_ohe

def run(label_class, batch_size, config_name, num_of_generations,
                    cross_validation_index, num_class, X_train, Y_train):
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, '../config//')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path+config_name)

    print(f"Running submodel training for fold #{cross_validation_index}, label #{label_class}")
    # Load profiling traces
    Y_train_ohe = ohe(Y_train=Y_train, label_class=label_class, num_class=num_class)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)
    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run for up to 300 generations.
    pe = neat.ParallelEvaluator(8, eval_genome)
    winner, stop_evol1, stop_evol2 = p.run(pe.evaluate, num_of_generations, batch_size,
                                           X_train, Y_train_ohe)
    # Display the winning genome.
    print('\nBest genome:{!s} Fitness: {!r}'.format(winner.key, winner.fitness))
    print('\nStopped by: Step1:{!s}, Step2:{!s}'.format(stop_evol1, stop_evol2))
    return winner.net



