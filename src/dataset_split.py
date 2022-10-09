import os
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold

AES_Sbox_inv = np.array([0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38,
     0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
     0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87,
     0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
     0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d,
     0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
     0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2,
     0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
     0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16,
     0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
     0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda,
     0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
     0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a,
     0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
     0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02,
     0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
     0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea,
     0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
     0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85,
     0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
     0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89,
     0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
     0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20,
     0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
     0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31,
     0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
     0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d,
     0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
     0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0,
     0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
     0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26,
     0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
])

def labels(metadata):
    ciphertext = metadata['ciphertext']
    key = metadata['key']
    key_last_byte = [i[15] for i in key]
    ciphertext_twelfth_byte = [i[11] for i in ciphertext]
    ciphertext_last_byte = [i[15] for i in ciphertext]
    label_first_part = [AES_Sbox_inv[int(i) ^ int(j)] for i, j in zip(key_last_byte, ciphertext_last_byte)]
    label = [int(i)^int(j) for i,j in zip(label_first_part, ciphertext_twelfth_byte)]
    return np.asarray(label)

def batch_data_ovr(x, y, md, m): ### added by Rabin with addition of metadata (md)
    x_batch = list()
    y_batch = list()
    md_batch = list()
    if type(x) is not list and type(y) is not list:
        x = x.tolist()
        y = y.tolist()
        md = md.tolist()
    for j in range(0,256):
        i = 0
        for a, b, c in zip(x, y, md):
            if b == j and i < m:
                x_batch.append(a)
                y_batch.append(b)
                md_batch.append(c)
                i += 1
            if len(x_batch) == m * 256:
                break
    return x_batch, y_batch, md_batch


def data_split(in_file, dataset_name, num_folds):
    dataset_path = os.getcwd() + "\\" + dataset_name
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    metadata = np.array(in_file['Profiling_traces/metadata'])
    # Load profiling traces
    X_profile = np.array(in_file['Profiling_traces/traces'],
                         dtype=np.int16)  ##changed to np.int16 for variable key dataset

    ##normalize X_profile
    X_profile_normalized = (X_profile - np.min(X_profile)) / (np.max(X_profile) - np.min(X_profile))
    # Load attacking labels
    if dataset_name == 'AES_HD':
        Y_profile = labels(metadata)
    else:
        Y_profile = np.array(in_file['Profiling_traces/labels'])

    ## number of each labels figured out below
    frequency_labels = list()
    for i in range(0, 256):
        frequency_labels.append((Y_profile == i).sum())
    minimum_frequency = min(frequency_labels)

    ### get the closest number divisible by 10 to prepare for 10 folds validation
    frequency_final = minimum_frequency - (minimum_frequency % num_folds)
    ### create a balanced dataset with all labels having same frequency equal to frequency_final
    X, y, md = batch_data_ovr(X_profile_normalized, Y_profile, metadata, frequency_final)

    ## check if the dataset is balanced
    for i in range(0, 256):
        if (y.count(i) != frequency_final):
            print('Error: dataset not balanced')
            break
    X, y, md = np.asarray(X), np.asarray(y), np.asarray(md)

    kfold = StratifiedKFold(n_splits=num_folds, shuffle=False, random_state=None)
    kfold.get_n_splits(X, y)
    # enumerate the splits and summarize the distributions
    i = 1
    for train_ix, test_ix in kfold.split(X, y):
        # select rows
        train_X, test_X = X[train_ix], X[test_ix]
        train_y, test_y = y[train_ix], y[test_ix]
        train_md, test_md = md[train_ix], md[test_ix]
        with open(dataset_name + '//train_X_normalized_' + str(i) + '.csv', 'w') as FOUT_train:
            np.savetxt(FOUT_train, train_X)
        with open(dataset_name + '//train_X_for_stacked_normalized_' + str(i) + '.csv', 'w') as FOUT_test:
            np.savetxt(FOUT_test, test_X)
        with open(dataset_name + '//train_y_normalized_' + str(i) + '.csv', 'w') as FOUT_trainy:
            np.savetxt(FOUT_trainy, train_y)
        with open(dataset_name + '//train_y_for_stacked_normalized_' + str(i) + '.csv', 'w') as FOUT_testy:
            np.savetxt(FOUT_testy, test_y)
        with open(dataset_name + '//train_md_normalized_' + str(i) + '.txt', 'wb') as FOUT_trainmd:
            pickle.dump(train_md, FOUT_trainmd)
        with open(dataset_name + '//train_md_for_stacked_normalized_' + str(i) + '.txt', 'wb') as FOUT_testmd:
            pickle.dump(test_md, FOUT_testmd)
        print("Creating fold " + str(i) + " for " + dataset_name + " dataset.")
        i += 1
    print("Folds of the dataset saved at:\n" + os.getcwd() + "\\" + dataset_name)


