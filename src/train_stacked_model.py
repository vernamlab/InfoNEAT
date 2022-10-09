import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression

class train_stacked_model:
    def __init__(self, dataset_name, X_stacked, Y_stacked, stacked_batch_size, cross_validation_index, num_class, sub_models):
        self.dataset_name = dataset_name
        self.X_stacked = X_stacked
        self.Y_stacked = Y_stacked
        self.stacked_batch_size = stacked_batch_size
        self.cross_validation_index = cross_validation_index
        self.num_class = num_class
        self.sub_models = sub_models

    def batch_data(self):
        m = self.stacked_batch_size
        x = self.X_stacked
        y = self.Y_stacked
        sss = StratifiedShuffleSplit(n_splits=1, test_size = m)
        x_batch = list()
        y_batch = list()
        for i, j in sss.split(np.zeros(len(y)), y):
            for k in j:
                x_batch.append(x[k])
                y_batch.append(y[k])
        return x_batch, y_batch

    # create stacked model input dataset as outputs from the ensemble
    def stacked_dataset(self, inputX):
        stackX = None
        print("Creating a stacked dataset with submodel predictions.")
        for net in self.sub_models:
            # make prediction
            yhat = list()
            for xi in inputX:
                yhat.append(net.activate(xi))
            if stackX is None:
                stackX = yhat
            else:
                stackX = np.dstack((stackX, yhat))
        # flatten predictions to [rows, members x probabilities]
        stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
        return stackX

    def fit_stacked_model(self, inputX, inputy):
        # create dataset using ensemble
        stackedX = train_stacked_model.stacked_dataset(self, inputX)
        # fit standalone model
        print("Fitting the stacked dataset against a logistic regression model.")
        model = LogisticRegression(multi_class='ovr')
        model.fit(stackedX, inputy)
        self.stacked_model = model

    def run(self):
        X_train, Y_train = train_stacked_model.batch_data(self)
        train_stacked_model.fit_stacked_model(self, X_train, Y_train)



