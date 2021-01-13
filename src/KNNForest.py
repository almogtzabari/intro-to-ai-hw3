import argparse

from utils import *
from Classifier import Classifier
from sklearn.model_selection import KFold
from DecisionTree import DecisionTreeClassifier
from sklearn.utils.random import sample_without_replacement
from sklearn.metrics.pairwise import euclidean_distances
from ID3 import ID3


__doc__ = \
    """
DESCRIPTION:
    Running this file will fit a KNNForest classifier and print the accuracy (classification rate) on the test-set.
    You can choose whether to find best params (N, K, p) using 5-Cross-Validation (use -find [NUM_PARAMS_TO_TRY]) or use
    params found earlier by Almog.  
"""


class KNNForestClassifier(Classifier):
    def __init__(self, N, K, p):
        super().__init__()
        if K > N:
            raise Exception(f"K cannot be bigger than N!")
        self.N = N
        self.K = K
        self.p = p
        self.trees = []
        self.centroids = None
        self._dtype = None

    def fit(self, X, y):
        if len(X) != len(y):
            raise Exception(f"Dimension Error! len(X) != len(y).")
        if len(y) <= 0:
            raise Exception(f"Cannot fit model without examples!")
        self._dtype = type(y[0])

        # Build trees
        centroids = []
        for tree_id in range(self.N):
            # Choose train indices
            indices = sample_without_replacement(n_population=len(X), n_samples=int(self.p * len(X)), random_state=ID)
            self.trees.append(DecisionTreeClassifier().use_alg(ID3).fit(X[indices], y[indices]))
            centroids.append(X[indices].mean(axis=0))
        self.centroids = np.array(centroids)
        return self

    def predict(self, X):
        if len(self.trees) == 0:
            raise Exception(f"You have to fit the model first!")
        closest_trees_per_sample = self._get_closest_trees(X)
        results = np.empty((len(X), len(self.trees)), dtype=self._dtype)
        y_hat = np.empty(len(X), dtype=self._dtype)
        for tree_id, tree in enumerate(self.trees):
            samples_for_this_tree = np.where(closest_trees_per_sample == tree_id)[
                0]  # Choose all samples that need to be predicted on this tree
            if len(samples_for_this_tree) > 0:
                tree_results = tree.predict(X[samples_for_this_tree])
                results[samples_for_this_tree, tree_id] = tree_results
        y_hat[np.sum(results == "B", axis=-1) >= np.sum(results == "M", axis=-1)] = "B"  # Majority vote
        y_hat[np.sum(results == "M", axis=-1) > np.sum(results == "B", axis=-1)] = "M"  # Majority vote
        return y_hat

    def _get_closest_trees(self, X):
        """ Returns K closest trees for each sample in X."""
        distances = euclidean_distances(X, self.centroids)  # Distances from each sample to all trees
        ordered_trees_id = np.argsort(distances)[:, :self.K]  # indices of K closest trees
        # return np.take(self.trees, ordered_trees_id)
        return ordered_trees_id


def find_best_params(num_trails: int, return_score: bool = False):
    avg_score = []
    X_train, y_train = get_dataset(DataSet.TRAIN_SET)

    # Split indices
    train_indices = []
    val_indices = []
    for train_idx, val_idx in KFold(n_splits=N_SPLITS, shuffle=True, random_state=ID).split(X_train):
        train_indices.append(train_idx)
        val_indices.append(val_idx)

    # Get random params (N, K, p)
    params = [get_random_params_for_knn_forest() for _ in range(num_trails)]

    # Run K-Fold-Cross-Validation
    for trail_id, (N, K, p) in enumerate(params):
        print(f"Running trail {trail_id + 1} out of {num_trails} for params (N, K, p) = ({N}, {K}, {p:.2f})")
        score = []
        for i in range(len(train_indices)):
            # Train model with value M
            train_idx = train_indices[i]
            curr_X_train = X_train[train_idx]
            curr_y_train = y_train[train_idx]
            knn_fdt = KNNForestClassifier(N, K, p).fit(curr_X_train, curr_y_train)

            # Validate
            val_idx = val_indices[i]
            curr_X_val = X_train[val_idx]
            curr_y_val = y_train[val_idx]
            curr_y_hat = knn_fdt.predict(curr_X_val)
            score.append(classification_rate(curr_y_hat, curr_y_val))
            print(f"Validation {i + 1} out of {len(train_indices)}, score: {score[-1]}")
        avg_score.append(np.average(score))
        print(f"Average validation score for these params: {np.average(score)}\n")

    best_params = params[np.argmax(avg_score)]
    print(f"Best params found: (N, K, p) = ({best_params[0]}, {best_params[1]}, {best_params[2]:.2f})")
    print(f"Average score for best params is: {np.max(avg_score)}")
    if return_score:
        return best_params, avg_score
    else:
        return best_params


def main(args=None):
    def get_parser():
        """
        Creates a new argument parser.
        """
        _parser = argparse.ArgumentParser(
            'KNNForest',
            formatter_class=argparse.RawTextHelpFormatter,
            description=__doc__,
        )
        _parser.add_argument('-find', '--find', type=int, default=None,
                             help=f"Find optimal params to use (N, K, p).\nUse `-find [NUM_PARAMS_TO_TRY]` to try "
                                  f"`NUM_PARAMS_TO_TRY`(int) different params.\n"
                                  f"Default are best found earlier by Almog: (N, K, P) = ({BEST_PARAMS[0]}, {BEST_PARAMS[1]}, {BEST_PARAMS[2]})\n"
                                  f"Finding best params is using randomly select params `NUM_PARAMS_TO_TRY` times and find the ones that "
                                  f"maximizing the average accuracy on the validation-set.\n"
                                  f"** NOTE ** : Some constants like N_MIN, N_MAX, etc. can be found at constants.py and you can change them as well."
                             )
        return _parser

    parser = get_parser()
    args = parser.parse_args(args)

    # Find best parameters
    if args.find is not None:
        N, K, p = find_best_params(num_trails=args.find)
    else:
        N, K, p = BEST_PARAMS

    # Train the model on entire train-set
    X_train, y_train = get_dataset(DataSet.TRAIN_SET)
    knn_fdt = KNNForestClassifier(N, K, p).fit(X_train, y_train)

    # Evaluate the model
    X_test, y_test = get_dataset(DataSet.TEST_SET)
    y_hat = knn_fdt.predict(X_test)
    print(classification_rate(y_hat, y_test))


if __name__ == '__main__':
    main()
