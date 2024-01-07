import numpy as np


class LinearSVM:
    def __init__(self, learning_rate=0.01, C=1, epochs=100):
        self.learning_rate = learning_rate
        self.C = C
        self.epochs = epochs
        self.possible_outcomes = []
        self.weights = None
        self.bias = None

    def fit(self, X, y):

        # X_ = np.column_stack((np.ones(len(X)), X))
        self.possible_outcomes = np.unique(y)
        y_ = np.where(y == self.possible_outcomes[0], 1, -1)

        obs, features = X.shape
        # obs, features = X_.shape
        self.weights = np.zeros(features)
        self.bias = 0

        for epoch in range(self.epochs):
            for i in range(obs):
                # if y_[i] * np.dot(X_[i], self.weights) < 1:
                #     self.weights += self.learning_rate * (y_[i] * X_[i] - 2 * self.weights)

                if y_[i] * (X.iloc[i, :] @ self.weights + self.bias) < 1:
                    self.weights -= self.learning_rate * (2 * self.weights - self.C * y_[i] * X.iloc[i, :])
                    self.bias -= - self.learning_rate * self.C * y_[i]


    def predict(self, X):

        outcome_mapping = {1: self.possible_outcomes[0], -1: self.possible_outcomes[1]}
        result = np.sign(X @ self.weights + self.bias)

        # X_ = np.column_stack((np.ones(len(X)), X))
        # result = np.sign(np.dot(X_, self.weights))

        return result.map(outcome_mapping)
