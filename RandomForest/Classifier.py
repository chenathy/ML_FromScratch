import numpy as np
import pandas as pd
# from collections import Counter


from DecisionTree import Classifier
from StatsFuncs.statistics import Statistics



class RandomForest:

    def __init__(self, n_trees=25, min_sample_split=2, max_depth=5, criteria='gini'):
        self.n_trees = n_trees
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.criteria = criteria

        # Store individually trained decision tree
        self.decision_trees = []



    def fit(self, X, y):

        # Forest cleanse
        if len(self.decision_trees) > 0:
            self.decision_trees = []

        # Build individual trees
        for i in range(self.n_trees):
            try:
                print(f'Building Decision Tree #{i + 1} {"=" * 50}')
                X_sampled, y_sampled = Statistics.bootstrap_sample(X, y)
                clf = Classifier.DecisionTree(
                    criteria=self.criteria,
                    max_depth=self.max_depth,
                    min_sample_split=self.min_sample_split
                )
                clf.fit(X_sampled, y_sampled)
                self.decision_trees.append(clf)
            except Exception as e:
                print(f'Error occurred for train a DecisionTree: {e}')


    def predict(self, X):

        # Predict with every tree in the forest
        predictions = np.zeros((X.shape[0], len(self.decision_trees)), dtype='U50')
        for i, tree in enumerate(self.decision_trees):
            predictions[:, i] = tree.predict(X)

        # Final predict with majority voting (Bagging)
        y_pred = np.apply_along_axis(
            # lambda x: Counter(x).most_common(1)[0][0],
            lambda x: np.array(max(x, key=list(x).count), dtype='U50'),
            axis=1,
            arr=predictions
        )
        # return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=predictions)

        return y_pred
