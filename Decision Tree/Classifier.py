# Load Modules
import numpy as np
import pandas as pd
from Node import Node


class DecisionTree:

    def __init__(self, criteria='gini', max_depth=None, min_sample_split=2):
        """
        Initializes a decision tree with a specified splitting criterion,
        maximum depth and minimum number of samples required to split an internal node.
        """
        self.criteria = criteria
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.root = None


    @staticmethod
    def entropy(class_labels):
        _, counts = np.unique(class_labels, return_counts=True)
        probs = counts / np.sum(counts)
        log2_probs = np.log2(probs)
        return -np.sum(probs * log2_probs)


    @staticmethod
    def gini_impurity(class_labels):
        _, counts = np.unique(class_labels, return_counts=True)
        probs = counts / np.sum(counts)
        return 1 - np.sum(probs ** 2)

    def most_common_label(self, y):
        # return np.bincount(y).argmax()
        return max(y, key=list(y).count)


    def _information_gain(self, parent_label, child_label_list, criteria='gini'):
        """
        Calculates the information gain of a split based on the specified criterion.
        """

        child_weights = np.array([len(child_label) for child_label in child_label_list])
        assert len(parent_label) == np.sum(child_weights)
        child_weights = child_weights / len(parent_label)

        if criteria == 'entropy':

            parent_entropy = self.entropy(parent_label)
            child_entropy_list = np.array([self.entropy(child_label) for child_label in child_label_list])
            child_entropy_total = np.sum(child_weights * child_entropy_list)

            return parent_entropy - child_entropy_total

        elif criteria == 'gini':

            parent_gini = self.gini_impurity(parent_label)
            child_gini_list = np.array([self.gini_impurity(child_label) for child_label in child_label_list])
            child_gini_total = np.sum(child_weights * child_gini_list)

            return parent_gini - child_gini_total

        else:
            print(f'Measurement {criteria} hasn\'t been created in Inforamation Gain yet.')
            pass

    def _best_split(self, X, y):

        best_feature = None
        best_threshold = None
        best_gain = -1

        for feature in X.columns:
            # Check if feature is a numeric one
            if np.issubdtype(X.loc[:, feature].dtype, np.number):
                thresholds = np.unique(X.loc[:, feature])
                for threshold in thresholds:
                    gain = self._information_gain(
                        y,
                        [y[X.loc[:, feature] <= threshold], y[X.loc[:, feature] > threshold]],
                        self.criteria
                    )

                    if gain >= best_gain:
                        best_feature, best_threshold, best_gain = feature, threshold, gain

            else:

                gain = self._information_gain(
                    y,
                    [y[X.loc[:, feature] == value] for value in set(X.loc[:, feature])],
                    self.criteria
                )

                if gain > best_gain:
                    best_feature, best_gain = feature, gain

        return best_feature, best_threshold, best_gain


    def _build_tree(self, X, y, depth=0):

        """recursive function to build the tree"""
        num_samples, num_features = X.shape
        num_classes = len(np.unique(y))

        # Stopping Criteria
        if (self.max_depth is not None and depth >= self.max_depth) \
            or (num_samples < self.min_sample_split) \
            or (num_classes != 1):

            print('Start splitting the tree at depth {depth}...')
            best_feature_, best_threshold_, best_gain_ = self._best_split(X, y)
            print(f'best gain value: {best_gain_}')
            print(f'best feature: {best_feature_}')
            if best_gain_ > 0:

                # Recursively build the left and right subtrees
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

                # find the best split node
                return Node(
                    feature=best_feature_,
                    threshold=best_threshold_,
                    left=left_tree,
                    right=right_tree,
                    info_gain=best_gain_
                )

            else:
                print('Tree splitting stopped, leaf node returned...')
                return Node(value=self.most_common_label(y))

        else:
            print('Tree splitting stopped, leaf node returned...')
            return Node(value=self.most_common_label(y))





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




    def print_tree(self, tree=None, indent= '    '):

        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print(f'{tree.feature} <= {tree.threshold} (info_gain: {tree.info_gain:.2f})')
            print(f'{indent}left:', end='')
            self.print_tree(tree.left, indent + indent)
            print(f'{indent}right:', end='')
            self.print_tree(tree.right, indent + indent)



