from Classifier import Classifier
import numpy as np


class DecisionTreeClassifier(Classifier):
    class DecisionTreeNode:
        def __init__(self, feature_name, feature_threshold, sub_tress, label):
            self.feature = feature_name
            self.threshold = feature_threshold
            self.sub_trees = sub_tress
            self.label = label

        def predict(self, X: np.ndarray):
            if len(self.sub_trees) == 0:
                return np.full(len(X), self.label)

            smaller_indices = (X.T[self.feature] < self.threshold)
            smaller_res = self.sub_trees[0].predict(X[smaller_indices])

            bigger_equal_indices = (X.T[self.feature] >= self.threshold)
            bigger_equal_res = self.sub_trees[1].predict(X[bigger_equal_indices])

            # Merge results
            y_hat = np.empty(len(X), dtype=type(self.label))
            y_hat[smaller_indices] = smaller_res
            y_hat[bigger_equal_indices] = bigger_equal_res
            return y_hat

    def __init__(self):
        super().__init__()
        self.root = None
        self.alg_fn = None

    def use_alg(self, alg_fn: callable):
        """
        Set an algorithm to create the decision tree accordingly.
        The function must accept two inputs:
        1. X: 2D matrix of features.
        2. y: 1D vector of labels (ground truth).
        """
        self.alg_fn = alg_fn
        return self

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the model using training set.
        :param X: 2D matrix of features.
        :param y: 1D vector of labels (ground truth).
        :return: The decision tree created.
        """
        if self.alg_fn is None:
            raise Exception(f"You must provide training algorithm first using `use_alg` method.")
        self.root = self.alg_fn(X, y)
        return self

    def predict(self, X: np.ndarray):
        """
        Predict a classification
        :param X: 2D matrix of features.
        :return: 1D vector of predictions
        """
        if self.root is None:
            raise Exception(f"You must fit the model first!")
        return self.root.predict(X)
