import csv
import numpy as np
from numpy import linalg as LA
import math
import neat
import os
from neat.six_util import itervalues
from neat.graphs import feed_forward_layers
from sklearn.utils import shuffle

def batch_data_ovr(xdata, ydata, m): ### added by Rabin
    x_batch = list()
    y_batch = list()
    x, y = shuffle(xdata, ydata)
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

# MNIST_inputs_complete = list(csv.reader(open('train_x.csv')))
# MNIST_outputs_complete = list(csv.reader(open('C:/Users/rabin/Downloads/neat-python-0.92/examples/xor/data/train_y_5vsRest.csv')))
# MNIST_inputs_complete = [[float(y)/255 for y in x] for x in MNIST_inputs_complete]
# MNIST_outputs_complete = [[float(y) for y in x] for x in MNIST_outputs_complete]
# MNIST_inputs, MNIST_outputs = batch_data_ovr(MNIST_inputs_complete, MNIST_outputs_complete, 100)

def guassianMatrix(X, sigma):

    X = np.array(X)
    X = X.astype(np.float)
    X_p = np.transpose(X)

    G = np.matmul(X, X_p)
    K = np.subtract(2 * G, np.transpose(np.diag(G)))
    K = np.exp((1/(2*sigma**2))*(np.subtract(K, np.diag(G))))
    return K

def MI_silverman(var1, var2, sigma1, sigma2, alpha):
    # estimate entropy
    input = np.array(var1)
    # input = input[:,1:len(var1[0])]
    K_x = (guassianMatrix(input, sigma1)/np.size(input, 0)).real
    L_x, tilde1 = LA.eig(K_x)
    lambda_x = np.absolute(np.diag(np.diag(L_x)))
    H_x = (1/(1-alpha))*math.log(np.sum(np.power(lambda_x, alpha)))

    target = np.array(var2)
    # target = target[:,1:len(var2[0])]
    K_y = (guassianMatrix(target, sigma2)/np.size(input, 0)).real
    L_y, tilde2 = LA.eig(K_y)
    lambda_y = np.absolute(np.diag(np.diag(L_y)))
    H_y = (1/(1-alpha))*math.log(np.sum(np.power(lambda_y, alpha)))

    # estimate joint entropy
    K_xy = np.multiply(K_x, K_y)*np.size(input, 0)
    L_xy, tilde12 = LA.eig(K_xy)
    lambda_xy = np.absolute(np.diag(np.diag(L_xy)))
    H_xy = (1/(1-alpha))*math.log(np.sum(np.power(lambda_xy, alpha)))

    # mutual information estimation
    mutual_information = H_x + H_y - H_xy
    return mutual_information

def CMI_estimation(var1, var2, labels, sigma1, sigma2, sigma_labels, alpha):
    # estimate entropy
    input = np.array(var1)
    # input = input[:,1:len(var1[0])]
    K_var1 = (guassianMatrix(input, sigma1)).real/np.size(input, 0)

    lab = np.array(labels)
    # lab = lab[:,1:len(labels[0])]
    K_lab = (guassianMatrix(lab, sigma_labels).real)/np.size(input, 0)

    target = np.array(var2)
    # target = target[:,1:len(var2[0])]
    K_var2 = (guassianMatrix(target, sigma2)/np.size(input, 0)).real


    L_y = LA.eigvals(K_var2)
    # lambda_y = np.absolute(np.diag(L_y))
    lambda_y = np.absolute(L_y)
    temp_arg = np.sum(np.power(lambda_y, alpha))
    # if temp_arg == 0.0:
    #     temp_arg = 1e-5
    H_var2 = (1/(1-alpha))*math.log(temp_arg)

    # estimate joint entropy
    K_xy = np.multiply(K_var1, K_var2)*np.size(input, 0)
    L_xy = LA.eigvals(K_xy)
    # lambda_xy = np.absolute(np.diag(L_xy))
    lambda_xy = np.absolute(L_xy)
    temp_arg = np.sum(np.power(lambda_xy, alpha))
    # if temp_arg == 0.0:
    #     temp_arg = 1e-5
    H_var1var2 = (1/(1-alpha))*math.log(temp_arg)

    #H(label,var2)
    K_labvar2 = np.multiply(K_lab, K_var2)*np.size(input, 0)
    L_labvar2 = LA.eigvals(K_labvar2)
    lambda_labvar2 = np.absolute(L_labvar2)
    temp_arg = np.sum(np.power(lambda_labvar2, alpha))
    # if temp_arg == 0.0:
    #     temp_arg = 1e-5
    H_labvar2 = (1 / (1 - alpha)) * math.log(temp_arg)

    #H(var1,label,var2)
    K_var1labvar2 = np.multiply(K_var1, K_labvar2)*np.size(input, 0)
    L_var1labvar2 = LA.eigvals(K_var1labvar2)
    lambda_var1labvar2 = np.absolute(L_var1labvar2)
    temp_arg = np.sum(np.power(lambda_var1labvar2, alpha))
    # if temp_arg == 0.0:
    #     temp_arg = 1e-5
    H_var1labvar2 = (1 / (1 - alpha)) * math.log(temp_arg)

    # mutual information estimation
    CMI = H_var1var2 + H_labvar2 - H_var1labvar2 - H_var2
    return CMI

def nnanalysis(MNIST_inputs, alpha, layers, nn_a):
    # nn analysis excluded
    # mutual information estimation in each layer

    nn_size = [len(l) for l in layers] # nn_size = [784, 256, 100, 36, 100, 256, 784] ##number of nodes in each layer
    n = len(layers) # n = 7 ## number of layers
    m = len(MNIST_inputs) # m = 100 ## number of samples

    mutual_information = np.zeros((2, n-2))
    for i in range(1, n-1):
        sigma1 = 5*m**(-1/(4+nn_size[0]))
        sigma2 = 5*m**(-1/(4+nn_size[i]))
        if i == ((n-1)/2):
            sigma2 = 5*sigma2
        mutual_information[0][i-1] = MI_silverman(nn_a[0], nn_a[i], sigma1, sigma2, alpha)
        mutual_information[1][i-1] = MI_silverman(nn_a[i], nn_a[n-1], sigma1, sigma2, alpha)
    num_symmetric_layers = int((n+1)/2)
    MI_symmetric = np.zeros(num_symmetric_layers)
    for j in range(0, num_symmetric_layers):
        sigma = m**(-1/(4+nn_size[j]))
        if j == num_symmetric_layers-1:
            sigma = 5 * sigma
        MI_symmetric[j] = MI_silverman(nn_a[j], nn_a[n-j-1], sigma, sigma, alpha)
    return mutual_information, MI_symmetric
# def nodes_input_output_by_layer():
    # restoring population from the checkpoint

def nnanalysis_CMI(MNIST_outputs, alpha, layers_a, layers_b, nn_a, nn_b):
    # nn analysis excluded
    # mutual information estimation in each layer

    nn_size_a = [len(l) for l in layers_a] # nn_size = [784, 256, 100, 36, 100, 256, 784] ##number of nodes in each layer
    nn_size_b = [len(l) for l in layers_b]

    n = len(layers_a) # n = 7 ## number of layers
    n_temp = len(layers_b)

    if n - n_temp > 0:
        n = n_temp

    m = len(MNIST_outputs) # m = 100 ## number of samples

    sigma_labels = 5*len(MNIST_outputs)**(-1/(4+len(MNIST_outputs[0])))
    mutual_information = list()
    for i in range(n):
        sigma1 = 5*m**(-1/(4+nn_size_a[i]))
        sigma2 = 5*m**(-1/(4+nn_size_b[i]))
        if i == ((n+1)/2):
            sigma2 = 5*sigma2
        mutual_information.append(CMI_estimation(nn_a[i], nn_b[i], MNIST_outputs, sigma1, sigma2, sigma_labels, alpha))

    return mutual_information

def nnanalysis_CMI_allnodes(MNIST_inputs, MNIST_outputs, alpha, nn_a, nn_b):
    # nn analysis excluded
    # mutual information estimation in each layer

    ## concatentate all outputs for all the nodes in the genome into one
    nn_a_all = list()
    nn_b_all = list()
    for i in range(len(MNIST_outputs)):
        nn_a_each = list()
        nn_b_each = list()
        for j in nn_a:
            nn_a_each.append(nn_a[j][i])
        nn_a_each_concat = [a for b in nn_a_each for a in b]
        nn_a_all.append(nn_a_each_concat)
        for j in nn_b:
            nn_b_each.append(nn_b[j][i])
        nn_b_each_concat = [a for b in nn_b_each for a in b]
        nn_b_all.append(nn_b_each_concat)

    m = len(MNIST_inputs) # m = 100 ## number of samples

    sigma_labels = 5*len(MNIST_outputs)**(-1/(4+len(MNIST_outputs[0])))
    mutual_information = list()

    sigma1 = 5*m**(-1/(4+len(nn_a_all[0])))
    sigma2 = 5*m**(-1/(4+len(nn_b_all[0])))

    mutual_information.append(CMI_estimation(nn_a_all, nn_b_all, MNIST_outputs, sigma1, sigma2, sigma_labels, alpha))

    return mutual_information

def outputs_nodes(MNIST_inputs, net, layers):
    nn_a = {}
    # for x in MNIST_inputs:
        # net.activate(x)
    for i,j in enumerate(layers):
        nn_a_node = list()
        nn_a_node_values = list()
        for y in j:
            nn_a_node_values.append(net.values[y])
        nn_a_node.append(nn_a_node_values)
        if i in nn_a.keys():
            nn_a[i].append(nn_a_node)
        else:
            nn_a[i] = nn_a_node

    for i in nn_a:
        for j, k in enumerate(nn_a[i]):
            if j > 0:
                nn_a[i][j] = nn_a[i][j][0]
    return nn_a

# local_dir = os.path.dirname(__file__)
# config_file = os.path.join(local_dir, 'config-feedforward-exp1')
# config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
#                      neat.DefaultSpeciesSet, neat.DefaultStagnation,
#                      config_file)
# p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-5vsRest')
# # for i in p.population:
# net = neat.nn.FeedForwardNetwork.create(p.population[790], config)
# connections = [cg.key for cg in itervalues(p.population[790].connections) if cg.enabled]
# layers = feed_forward_layers(config.genome_config.input_keys, config.genome_config.output_keys, connections)
# layers.insert(0, net.input_nodes)
# nn_a = outputs_nodes(MNIST_inputs, net, layers)
# mutual_information, MI_symmetric = nnanalysis(1.01, layers, nn_a)
# mutual_information_CMI = nnanalysis_CMI(1.01, layers, nn_a, MNIST_outputs)


