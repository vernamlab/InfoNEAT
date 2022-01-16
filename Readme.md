################## SCA using InfoNEAT ###############

1. For each dataset, run data_split_aeshd.py to create a balanced dataset of 10 folds. As an example,  AES_HD dataset is attacked using InfoNEAT. The AES_HD dataset and instructions can be found in https://github.com/AISyLab/AES_HD.

2. Run ae_neat_one_vs_rest.py to train 256 submodels.

3. Run stacked_model_cross_validation.py to train a stacked submodel.

4. Run meta_learner_cross_validation.py that uses the submodels and the stacked submodes to obtain the rank files which are saved as rank_x, rank_y text files.

Result:
1. 10-folds of train_X, train_y, train_md (metadata of each trace), test_X, test_y, test_md are generated in the folder AES_HD.

2. For each fold, 256 submodels are created in each cross-validation folder (1,2,3,...10). (E.g. for 1st fold, under models_aeshd/1/ folder, there will be 256 submodels generated). For storage reason, one submodel for the 1st fold is saved).

3. For each fold, a stacked model is generated. (stacked_model_aeshd_20_1 will be the submodes for 1st fold (not uploaded because of storage reasons)).

4. Then, rank files are generated in the folder ranks_aeshd (E.g. rank_x_1.txt and rank_y_1.txt)


###################################################
Note: modifications to the original neat code can be found in the folder 'neat'. Especially the code to train a submodel using batch sizes of data, training submodels for multiple classes or labels, and the inclusion of CMI criteria to train effective submodel)