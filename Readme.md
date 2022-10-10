InfoNEAT framework to train deep models that performs SCA

Python packages to install: scikit-learn, h5py, pickle, csv

Instructions: 

1. Extract the files from the folder 'InfoNEAT'.

2. The folder contains a config folder which contains configuration parameters for the NEAT based algorithm for each dataset. The src folder contains all the source code to train sub-models, stacked models, and even split the dataset into different cross-validation folds.

2. The file 'train.py' contains the code to train sub-models and stacked model for each dataset. Change lines 9-14 accordingly. If needed, set dataset_split = True (line 15) to create cross-folds of the dataset. Please save the .h5 dataset files for each of the dataset in the respective folder.

Result (after running the file 'train.py')
1. If dataset_split is set to True, then k-folds of the dataset will be created and saved under the datasets folder. (Currently num_of_folds is set to 5)

2. 256 sub-models and a stacked model for a specific fold will be created and saved under the models folder. (Currently fold_num is set to 1).

###################################################
Note: modifications to the original neat code can be found in the folder 'neat'. Especially the code to train a submodel using batch sizes of data, training submodels for multiple classes or labels, and the inclusion of CMI criteria to train effective submodel)
