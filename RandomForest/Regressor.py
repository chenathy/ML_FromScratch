import numpy as np

from DecisionTree import Regressor
from StatsFuncs.statistics import Statistics



class RandomForest:

    def __init__(self, n_trees=25, max_depth=5, min_sample_split=2):

        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split

        self.decision_trees = []


    def fit(self, X, y):

        # Forest clense
        if len(self.decision_trees) > 0 :
            self.decision_trees = []

        # Build individual tree
        for i in range(self.n_trees):
            try:
                print(f'Building Tree # {i + 1} {"=" * 50}')
                X_sampled, y_sampled = Statistics.bootstrap_sample(X, y)
                reg = Regressor.DecisionTree(
                    max_depth=self.max_depth,
                    min_sample_split=self.min_sample_split
                )
                reg.fit(X_sampled, y_sampled)
                self.decision_trees.append(reg)
            except Exception as e:
                print(e)
                pass

    def predict(self, X):
        # Predict with every individual tree
        predictions = np.zeros((X.shape[0], len(self.decision_trees)))
        for i, tree in enumerate(self.decision_trees):
            predictions[:, i] = tree.predict(X)

        # Final predict with average mean (Bagging)
        return np.mean(predictions, axis=1)

