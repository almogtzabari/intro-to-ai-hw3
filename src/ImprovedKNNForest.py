from utils import *
import argparse
from KNNForest import KNNForestClassifier


class ImprovedKNNForestClassifier(KNNForestClassifier):
    def __init__(self, N, K, p, T: float):
        super().__init__(N, K, p)
        self.T = T

    def _calc_weights(self, X: np.ndarray):
        trees_id_ordered_per_sample = self._get_tree_ids_sorted_by_distance(X)  # Find IDs of closest trees (per sample - this is a matrix)
        weights = np.full_like(trees_id_ordered_per_sample, fill_value=np.inf)
        weights[trees_id_ordered_per_sample < self.K] = trees_id_ordered_per_sample[trees_id_ordered_per_sample < self.K]  # Take only K closest trees
        enumerator = np.exp(-weights / self.T)
        denominator = np.expand_dims(np.sum(enumerator, axis=-1), axis=-1)
        weights = enumerator / denominator
        return weights


def find_best_params(num_trails: int, return_score: bool = False):
    print(f"Searching for best params for ImprovedKNNForestClassifier.")
    print(f"Looking for best (N, K, p, T).")
    print(f"Trying {num_trails} sets of different params.")
    # Get random params (N, K, p, T)
    params = [get_random_params_for_improved_knn_forest() for _ in range(num_trails)]

    avg_scores = []
    X_train, y_train = get_dataset(DataSet.TRAIN_SET)

    for trail_id, (N, K, p, T) in enumerate(params):
        print(f"\nRunning cross-validation for params set {trail_id + 1} out of {num_trails}.\nParams are (N, K, p, T) = ({N}, {K}, {p:.2f}, {T:.2f})")
        model = ImprovedKNNForestClassifier(N, K, p, T)
        avg_score = k_cross_validation(model, X_train, y_train, classification_rate)
        avg_scores.append(avg_score)
        print(f"Average validation score for these params: {np.average(avg_score)}\n")

    best_params = params[int(np.argmax(avg_scores))]
    print(f"Best params found: (N, K, p, T) = ({best_params[0]}, {best_params[1]}, {best_params[2]:.2f}, {best_params[3]:.2f})")
    print(f"Average score for best params is: {np.max(avg_scores)}")
    if return_score:
        return best_params, np.max(avg_scores)
    else:
        return best_params


def main(args=None):
    def get_parser():
        _parser = argparse.ArgumentParser(
            'ImprovedKNNForest',
            formatter_class=argparse.RawTextHelpFormatter,
            description=__doc__,
        )
        _parser.add_argument('-find', '--find', type=int, default=None,
                             help=f"Find optimal params to use (N, K, p, T).\nUse `-find [NUM_PARAMS_TO_TRY]` to try "
                                  f"`NUM_PARAMS_TO_TRY`(int) different params.\n"
                                  f"Default are best found earlier by Almog: (N, K, p, T) = ({BEST_IMPROVED_KNN_FOREST_PARAMS[0]}, {BEST_IMPROVED_KNN_FOREST_PARAMS[1]}, {BEST_IMPROVED_KNN_FOREST_PARAMS[2]}, {BEST_IMPROVED_KNN_FOREST_PARAMS[3]})\n"
                                  f"Finding best params is using randomly select params `NUM_PARAMS_TO_TRY` times and find the ones that "
                                  f"maximizing the average accuracy on the validation-set.\n"
                                  f"** NOTE ** : Some constants like N_MIN, N_MAX, etc. can be found at constants.py and you can change them as well."
                             )
        return _parser

    parser = get_parser()
    args = parser.parse_args(args)

    # Find best parameters
    if args.find is not None:
        N, K, p, T = find_best_params(num_trails=args.find)
    else:
        N, K, p, T = BEST_IMPROVED_KNN_FOREST_PARAMS

    # Train the model on entire train-set
    X_train, y_train = get_dataset(DataSet.TRAIN_SET)
    knn_fdt = ImprovedKNNForestClassifier(N, K, p, T).fit(X_train, y_train)

    # Evaluate the model
    X_test, y_test = get_dataset(DataSet.TEST_SET)
    y_hat = knn_fdt.predict(X_test)
    print(classification_rate(y_hat, y_test))


if __name__ == '__main__':
    main()