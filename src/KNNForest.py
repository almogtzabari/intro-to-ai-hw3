import argparse

from utils import *
from Classifier import Classifier
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
    n_classifiers = 0

    def __init__(self, N, K, p, use_weights: bool = False):
        super().__init__()
        if K > N:
            raise Exception(f"K cannot be bigger than N!")
        self.N = N
        self.K = K
        self.p = p
        self.trees = []
        self.centroids = None
        self._stds = None
        KNNForestClassifier.n_classifiers += 1

    def fit(self, X, y):
        if len(X) != len(y):
            raise Exception(f"Dimension Error! len(X) != len(y).")
        if len(y) <= 0:
            raise Exception(f"Cannot fit model without examples!")

        # Build trees
        centroids = []
        trees = []
        stds = []
        for tree_id in range(self.N):
            indices = sample_without_replacement(
                n_population=len(X),
                n_samples=int(self.p * len(X)),
                method="reservoir_sampling",
                random_state=ID + KNNForestClassifier.n_classifiers + tree_id
            )  # Choose train indices
            mean = X[indices].mean(axis=0)
            std = np.sqrt(np.mean(((X[indices] - mean) ** 2), axis=0))
            trees.append(DecisionTreeClassifier().use_alg(ID3).fit(std_normalization(X[indices], mean, std), y[indices]))
            centroids.append(mean)
            stds.append(std)

        self.centroids = np.array(centroids)
        self._stds = np.array(stds)
        self.trees = trees
        return self

    def predict(self, X):
        if len(X) == 0:
            return np.array([])
        if len(self.trees) == 0:
            raise Exception(f"You have to fit the model first!")

        # Calculate scores
        scores = []
        for tree_id, tree in enumerate(self.trees):
            scores.append(tree.predict(std_normalization(X, self.centroids[tree_id], self._stds[tree_id])))
        scores = np.array(scores).T  # For each sample there is a result for each tree (shape = [N_SAMPLES, N_TREES]).

        # Apply weights
        weights = self._calc_weights(X)
        weighted_scores = scores * weights

        # Classify based on sign
        y_hat = np.sign(np.sum(weighted_scores, axis=-1))
        y_hat[y_hat == 0] = 1
        return y_hat

    def _get_tree_ids_sorted_by_distance(self, X):
        """ For each sample returns the tree ids from closest to farthest. """
        distances = euclidean_distances(X, self.centroids)  # Distances from each sample to all trees
        return np.argsort(distances)

    def _calc_weights(self, X: np.ndarray):
        trees_id_ordered_per_sample = self._get_tree_ids_sorted_by_distance(X)  # Find IDs of closest trees (per sample - this is a matrix)
        weights = np.zeros_like(trees_id_ordered_per_sample)
        weights[trees_id_ordered_per_sample < self.K] = 1
        return weights


def find_best_params(num_trails: int, return_score: bool = False):
    print(f"Searching for best params for KNNForestClassifier.")
    print(f"Looking for best (N, K, p).")
    print(f"Trying {num_trails} sets of different params.")
    # Get random params (N, K, p)
    params = [get_random_params_for_knn_forest() for _ in range(num_trails)]

    avg_scores = []
    X_train, y_train = get_dataset(DataSet.TRAIN_SET)

    for trail_id, (N, K, p) in enumerate(params):
        print(
            f"\nRunning cross-validation for params set {trail_id + 1} out of {num_trails}.\nParams are (N, K, p) = ({N}, {K}, {p:.2f})")
        model = KNNForestClassifier(N, K, p)
        avg_score = k_cross_validation(model, X_train, y_train, classification_rate)
        avg_scores.append(avg_score)
        print(f"Average validation score for these params: {avg_score}\n")

    best_params = params[int(np.argmax(avg_scores))]
    print(f"Best params found: (N, K, p) = ({best_params[0]}, {best_params[1]}, {best_params[2]:.2f})")
    print(f"Average score for best params is: {np.max(avg_scores)}")
    if return_score:
        return best_params, np.max(avg_scores)
    else:
        return best_params


def main(args=None):
    def get_parser():
        _parser = argparse.ArgumentParser(
            'KNNForest',
            formatter_class=argparse.RawTextHelpFormatter,
            description=__doc__,
        )
        _parser.add_argument('-find', '--find', type=int, default=None,
                             help=f"Find optimal params to use (N, K, p).\nUse `-find [NUM_PARAMS_TO_TRY]` to try "
                                  f"`NUM_PARAMS_TO_TRY`(int) different params.\n"
                                  f"Default are best found earlier by Almog: (N, K, P) = ({BEST_KNN_FOREST_PARAMS[0]}, {BEST_KNN_FOREST_PARAMS[1]}, {BEST_KNN_FOREST_PARAMS[2]})\n"
                                  f"Finding best params is using randomly select params `NUM_PARAMS_TO_TRY` times and find the ones that "
                                  f"maximizing the average accuracy on the validation-set.\n"
                                  f"** NOTE ** : Some constants like N_MIN, N_MAX, etc. can be found at constants.py and you can change them as well."
                             )
        return _parser

    parser = get_parser()
    args = parser.parse_args(args)

    # Find best parameters
    if args.find is not None:
        find_best_params(num_trails=args.find)
    else:
        N, K, p = BEST_KNN_FOREST_PARAMS

        # Train the model on entire train-set
        X_train, y_train = get_dataset(DataSet.TRAIN_SET)
        knn_fdt = KNNForestClassifier(N, K, p).fit(X_train, y_train)

        # Evaluate the model
        X_test, y_test = get_dataset(DataSet.TEST_SET)
        y_hat = knn_fdt.predict(X_test)
        print(classification_rate(y_hat, y_test))


if __name__ == '__main__':
    main()
