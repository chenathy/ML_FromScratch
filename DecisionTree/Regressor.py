# Load Modules
import numpy as np
from DecisionTree.Node import Node


class DecisionTree:

    def __init__(self, max_depth=None, min_sample_split=2):

        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.root = None


    def _variance_reduction(self, parent, child_list):

        child_weights = np.array([len(child) for child in child_list])
        assert len(parent) == np.sum(child_weights)
        child_weights = child_weights / len(parent)

        # reduction
        return np.var(parent) - np.sum(child_weights * np.array([np.var(child) for child in child_list]))


    def _target_value(self, y):
        return np.mean(y)


    def _best_split(self, X, y):

        best_feature = None
        best_threshold = None,
        best_reduction = -1

        for feature in X.columns:
            # Check if feature is numeric one
            if np.issubdtype(X.loc[:, feature].dtype, np.number):
                thresholds = np.unique(X.loc[:, feature])
                for threshold in thresholds:
                    reduction = self._variance_reduction(
                        y,
                        [y[X.loc[:, feature] <= threshold], y[X.loc[:, feature] > threshold]]
                    )

                    if reduction > best_reduction:
                        best_feature, best_threshold, best_reduction = feature, threshold, reduction

            else:
                reduction = self._variance_reduction(
                    y,
                    [y[X.loc[:, feature] == value] for value in np.unique(X.loc[:, feature])]
                )

                if reduction > best_reduction:
                    best_feature, best_reduction = feature, reduction

        return best_feature, best_threshold, best_reduction


    def _build_tree(self, X, y, depth=0):

        num_samples, num_features = X.shape

        # Stopping Criteria
        if ((self.max_depth is not None and self.max_depth >= depth)
            and (self.min_sample_split < num_samples)):

            print(f'Splitting Tree at depth {depth}...')
            best_feature_, best_threshold_, best_reduction_ = self._best_split(X, y)
            if best_reduction_ > 0:

                # Recusive build left and right tree
                left_tree = self._build_tree(
                    X.loc[X.loc[:, best_feature_] <= best_threshold_, :],
                    y[X.loc[:, best_feature_] <= best_threshold_],
                    depth + 1
                )

                right_tree = self._build_tree(
                    X.loc[X.loc[:, best_feature_] > best_threshold_, :],
                    y[X.loc[:, best_feature_] > best_threshold_],
                    depth + 1
                )

                return Node(
                    feature=best_feature_,
                    threshold=best_threshold_,
                    left=left_tree,
                    right=right_tree,
                    info_gain=best_threshold_
                )

            else:
                print(f'Tree split stopped... leaf node returned')
                return Node(value=self._target_value(y))

        else:
            print('Tree didn\'t get to splitted.')
            return Node(value=self._target_value(y))



    def fit(self, X, y):
        self.root = self._build_tree(X, y)


    def _predict_row(self, row, node):
        if node.value != None: return node.value

        if row[node.feature] <= node.threshold:
            return self._predict_row(row, node.left)
        else:
            return self._predict_row(row, node.right)


    def predict(self, X):
        return [self._predict_row(row, self.root) for index, row in X.iterrows()]


    def print_tree(self, tree, indent = '  '):

        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print(f'{tree.feature} <= {tree.threshold} (reduction = {tree.info_gain:2f})')
            print(f'{indent}left:', end='')
            self.print_tree(tree.left, indent + indent)
            print(f'{indent}right:', end='')
            self.print_tree(tree.right, indent + indent)

