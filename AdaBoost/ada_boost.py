import numpy as np
from AdaBoost.decision_stump import *


class Ada_Boost:
    def __init__(self, n_rounds=5):
        self.n_rounds = n_rounds
        self.clfs = []


    def fit(self, X, y):
        """
        So in each round 𝑡:
            Loop through all features (columns):

            For each feature:

            * Loop through possible thresholds (from all rows of that feature)

            * Try both polarities (≤ or >)

            * Compute weighted error

            Pick the best (feature, threshold, polarity) that gives the lowest error

            Save that configuration as the decision stump for round 𝑡

        :param X:
        :param y:
        :return:
        """

        n_rows, n_features = X.shape
        weights = np.full(n_rows, 1/n_rows)

        for t in range(self.n_rounds):
            # Initialize Decision Stump
            clf = Decision_Stump()
            optimal_error = np.inf

            # Loop through all features
            for feature in range(n_features):
                thresholds = np.unique(X.iloc[:, feature])

                # Loop through all possible values of this feature
                for threshold in thresholds:

                    # Try both polarities (≤ or >)
                    for polarity in [-1, 1]:
                        predictions = np.ones(n_rows)
                        if polarity == 1:
                            predictions[X.iloc[:, feature] < threshold] = -1
                        else:
                            predictions[X.iloc[:, feature] > threshold] = -1

                        # Compute Weighted Error
                        error = np.sum(weights[predictions != y])
                        if error < optimal_error:
                            optimal_error = error
                            clf.threshold = threshold
                            clf.feature_index = feature
                            clf.polarity = polarity

            # Best Decision Stump detected after looping
            # Calculate Alpha (Weight of this Decision Stump)
            print(f'optimal error: {optimal_error}')

            clf.alpha = 0.5 * np.log((1 - optimal_error) / optimal_error)
            # eps = 1e-10  # avoid division by 0
            # clf.alpha = 0.5 * np.log((1.0 - optimal_error + eps) / (optimal_error + eps))
            print(f'Decision Stump {t + 1} ==> Feature index: {feature} ==> Alpha (weight) for this Decision: {clf.alpha:.3f}')

            # Update weights (of samples)
            # Next stump will focus more on those hard-to-classify samples in training.
            weights *= np.exp(-1 * clf.alpha * y * clf.predict(X))

            # Normalize updated weights
            sum_weights = np.sum(weights)
            weights /= sum_weights

            self.clfs.append(clf)

    def predict(self, X):
        """
        With `np.sign()` to determine the final predictions
        Result: array of only (-1, 1) as possible values in it
        """
        return np.sign(
            np.sum(
                [clf.alpha * clf.predict(X) for clf in self.clfs],
                axis=0
            )
        )





