import numpy as np
import pandas as pd

class MultinomialNB:

    def __init__(self, alpha=1):
        self.alpha = alpha
        self.classes = None
        self.class_counts = None
        self.feature_counts = None
        self.class_prior = None
        self.feature_prob = None

    def _multinomial_likelihood(self, X, feature_prob):
        return np.power(feature_prob, X)

    def fix(self, X, y):
        self.classes, self.class_counts = np.unique(y, return_counts=True)
        self.class_prior = self.class_counts / len(y)
        self.feature_counts = np.zeros((len(self.classes), X.shape[1]))
        self.feature_prob = np.zeros((len(self.classes), X.shape[1]))

        # Convert pd.DataFrame into np.array
        if isinstance(X, pd.DataFrame):
            X_ = X.to_numpy()
        else:
            X_ = X

        for i, c in enumerate(self.classes):
            _c = X_[ y == c, :]
            self.feature_counts[i, :] = np.sum(_c, axis=0)
            self.feature_prob[i, :] = (self.feature_counts[i, :] + self.alpha) / (np.sum(_c) + self.alpha * X.shape[1])

    def predict(self, X):
        likelihood = np.zeros((X.shape[0], len(self.classes)))

        for i, c in enumerate(self.classes):
            likelihood[:, i] = np.prod(
                self._multinomial_likelihood(
                    X,
                    self.feature_prob[i,:]
                ),
                axis=1
            )

        posterior = likelihood * self.class_prior

        pred_indexes = np.argmax(posterior, axis=1)

        return [self.class_prior[i] for i in pred_indexes]
