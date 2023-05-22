import numpy as np
import pandas as pd


class GaussianNB:

    def __init__(self):
        self.classes = None
        self.class_prior = None
        self.means_matrix = None # n_classes x n_features
        self.vars_matrix = None  # n_classes x n_features

    def _gaussian_pdf(self, X, mean, var):
        return  1 / np.sqrt(2 * np.pi * var) * np.exp(-((X - mean)) ** 2 / (2 * var))

    def fit(self, X, y):
        self.classes, class_count = np.unique(y, return_counts=True)
        self.class_prior = class_count / len(y)
        self.means_matrix = np.zeros((len(self.classes), X.shape[1]))
        self.vars_matrix = np.zeros((len(self.classes), X.shape[1]))

        # dataframe to array
        if isinstance(X, pd.DataFrame):
            X_ = X.to_numpy()
        else:
            X_ = X

        for i, c in enumerate(self.classes):
            _c = X_[y == c, :]
            self.means_matrix[i, :] = np.mean(_c, axis=0) # mean() on each feature
            self.vars_matrix[i, :] = np.var(_c, axis=0)   # var() on each feature

    def predict(self, X):
        likelihoods = np.zeros((X.shape[0], len(self.classes)))

        for i, c in enumerate(self.classes):
            likelihoods[:, i] = np.prod(
                self._gaussian_pdf(
                    X,
                    self.means_matrix[i, :],
                    self.vars_matrix[i, :]
                ),
                axis=1
            )

        ### No need to divided by p(X) to obtain posterior probability
        posterior = likelihoods * self.class_prior

        pred_index = np.argmax(posterior, axis=1)

        return [self.classes[i] for i in pred_index]
