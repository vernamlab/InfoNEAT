from src import *
import h5py
import pickle
from joblib import dump, load
from src.train_submodels import run
import numpy as np

if __name__ == '__main__':
    datasets_location = 'datasets//'
    dataset_name = 'AES_HD' ## possible values are AES_HD, ASCAD_fixed, ASCAD_variable
    dataset_file_name = 'aes_hd.h5' ### possible values are aes_hd.h5, ASCAD.h5, ASCAD_desync50.h5, ASCAD_desync100.h5
    submodels_location = 'models//submodels//' + dataset_name +'//'
    stacked_models_location = 'models//stacked_models//' + dataset_name +'//'
    num_of_folds = 5
    dataset_split = True

    ''''
    To create different cross-folds of the dataset, please set dataset_split = True
    '''
    if dataset_split == True:
        in_file = h5py.File(datasets_location + dataset_name + "/"+dataset_file_name, "r")
        data_split(in_file, dataset_name, num_of_folds)
    #########################################

    num_class = 256
    batch_size = 100
    config_name = 'config_'+dataset_name
    num_of_generations = 1
    fold_num = 1 ### vary this value to train submodels for k-fold cross-validation

    ######## training 256 submodels ##########################
    X_train = np.loadtxt(datasets_location + dataset_name+ "/train_X_normalized_"+str(fold_num)+".csv")
    Y_train = np.loadtxt(datasets_location + dataset_name+ "/train_Y_normalized_"+str(fold_num)+".csv")
    submodels = list()
    for label_class in range(0,num_class):
        submodel = run(label_class, batch_size, config_name, num_of_generations,
            fold_num, num_class, X_train, Y_train)
        with open(submodels_location +str(fold_num) + '//net-' + str(label_class), 'wb') as f:
            pickle.dump(submodel, f)
        submodels.append(submodel)
    #########################################

    ###### training a stacked submodel ##########################
    stacked_batch_size = 25 * num_class
    X_stacked = np.loadtxt(dataset_name + "//train_X_for_stacked_normalized_" + str(fold_num) + ".csv")
    Y_stacked = np.loadtxt(dataset_name + "//train_y_for_stacked_normalized_" + str(fold_num) + ".csv")
    train_stacked = train_stacked_model(dataset_name, X_stacked, Y_stacked, stacked_batch_size, fold_num, num_class,
                                        sub_models=submodels)
    train_stacked.run()
    stacked_model = train_stacked.stacked_model

    ### saves the stacked model under the models//stacked_models//dataset_name>//<fold_num> folder
    dump(stacked_model, stacked_models_location+str(fold_num)+'//stacked_'+dataset_name)
    #######################################
