InfoNEAT is a member of the NEAT (<a href="https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies">NeuroEvolution of Augmenting Topologies</a>) family. InfoNEAT evolves neural networks and knows when to stop learning and how to scale up as a multi-class algorithm with many (e.g., 256) classes. In a nutshell, two issues with the NEAT algorithm are resolved: application in multi-class scenarios and generic criteria for stopping the training/evolution processes. 

This repo presents the InfoNEAT framework used to train NN models that perform SCA. The results will appear in <a href="https://tches.iacr.org/index.php/TCHES/index">IACR Transactions on Cryptographic Hardware and Embedded Systems (TCHES)</a> (TCHES 2023, Issue 1). It is also available on <a href="https://arxiv.org/pdf/2105.00117.pdf">arxiv</a>.

Python packages to install: scikit-learn, h5py, pickle, csv

<b>Instructions:</b> <br>
<ul>

1. Extract the files from the folder 'InfoNEAT'.

2. The folder contains a config folder that contains configuration parameters for the NEAT-based algorithm for each dataset. The src folder contains all the source code to train sub-models, stacked models, and even split the dataset into different cross-validation folds.

3. The file 'train.py' contains the code to train sub-models and stacked model for each dataset. Change lines 9-14 accordingly. If needed, set dataset_split = True (line 15) to create cross-folds of the dataset. Please save the .h5 dataset files for each of the datasets under the respective folders.
  
</ul>

<b>Result (after running the file 'train.py'):</b> <br>
<ul>

1. If dataset_split is set to True, then k-folds of the dataset will be created and saved under the datasets folder. (Currently num_of_folds is set to 5)

2. 256 sub-models and a stacked model for a specific fold will be created and saved under the models folder (currently, fold_num is set to 1).
</ul>

Note: the models/submodels folder contains all the submodels against the three datasets. The models/stacked_models folder contains stacked models for one fold for AES_HD and ASCAD_fixed key datasets. This folder also contains the stacked models for the ASCAD_variable key dataset (no cross-validation was performed for the ASCAD_variable key dataset).

<br>###################################################<br>
Note: modifications to the original neat code can be found in the folder 'neat'. Especially the code to train a submodel using batch sizes of data, training submodels for multiple classes or labels, and the inclusion of CMI criteria to train effective submodel)
