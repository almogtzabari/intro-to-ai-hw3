import argparse
from utils import *
from ID3 import TDIDT
from DecisionTree import DecisionTreeClassifier


__doc__ = \
    """
DESCRIPTION:
    Running this file will fit a DecisionTreeClassifier with improved ID3 tuned for the specific loss function
    "ten_times_penalty". After fitting, the loss on the test-set will be printed.
    You can choose whether to find best epsilon using 5-Cross-Validation (use -find [NUM_PARAMS_TO_TRY]) or use
    epsilon found earlier by Almog.  
"""


def select_feature_based_on_max_information_gain_epsilon(X: np.ndarray, y: np.ndarray, epsilon: float):
    def _feature_ig(_X: np.ndarray, _y: np.ndarray, _feature, _threshold):
        def _calc_entropy(__y: np.ndarray):
            entropy = 0
            if len(__y) == 0:
                return entropy
            _, counts = np.unique(__y, return_counts=True)
            for count in counts:
                if count > 0:
                    p = count / len(__y)
                    entropy -= p * np.log2(p)
            return entropy

        bigger_equal_indices = (_X.T[_feature] >= _threshold)
        p_bigger_equal = np.average(bigger_equal_indices)
        smaller_indices = (_X.T[_feature] < _threshold)
        p_smaller = np.average(smaller_indices)
        return _calc_entropy(_y) - (p_bigger_equal * _calc_entropy(_y[bigger_equal_indices])) - (
                p_smaller * _calc_entropy(_y[smaller_indices]))

    def _calc_weighted_loss_for_feature_and_threshold(_f, _th):
        bigger_equal_indices = (X.T[_f] >= _th)
        smaller_indices = (X.T[_f] < _th)
        smaller_majority = 1 if np.sum(y[smaller_indices] == 1) >= np.sum(y[smaller_indices] == -1) else -1
        bigger_equal_majority = 1 if np.sum(y[bigger_equal_indices] == 1) >= np.sum(
            y[bigger_equal_indices] == -1) else -1
        smaller_loss = ten_times_penalty(np.full_like(y[smaller_indices], fill_value=smaller_majority),
                                         y[smaller_indices])
        bigger_equal_loss = ten_times_penalty(np.full_like(y[bigger_equal_indices], fill_value=bigger_equal_majority),
                                              y[bigger_equal_indices])
        loss = (len(y[smaller_indices]) / len(y)) * smaller_loss + (
                    len(y[bigger_equal_indices]) / len(y)) * bigger_equal_loss
        return loss

    max_ig = float("-inf")
    candidates = []
    for feature in range(X.shape[-1]):
        possible_values = np.unique(X.T[feature])
        thresholds = (possible_values[:-1] + possible_values[1:]) / 2
        for threshold in thresholds:
            ig = _feature_ig(X, y, feature, threshold)
            if ig >= max_ig:
                max_ig = ig
                candidates = list(filter(lambda cand: abs((cand[2] / max_ig) - 1) < epsilon, candidates))  # Remove irrelevant due to new best
            if abs((ig / max_ig) - 1) <= epsilon:
                candidates.append((feature, threshold, ig))

    if len(candidates) == 0:
        return None, None
    if len(candidates) == 1:
        f, th, _ = candidates[0]
        return f, th

    # Find best candidate based on "10 times loss"
    best_candidate_feature = None
    best_candidate_threshold = None
    best_loss = np.inf
    for f, th, _ in candidates:
        loss = _calc_weighted_loss_for_feature_and_threshold(f, th)
        if loss < best_loss:
            best_loss = loss
            best_candidate_feature = f
            best_candidate_threshold = th
    return best_candidate_feature, best_candidate_threshold


def cost_sensitive_id3(X: np.ndarray, y: np.ndarray, epsilon: float):
    c = calc_most_common_value(y)
    return TDIDT(X, y, c, select_feature_based_on_max_information_gain_epsilon, epsilon=epsilon)


def find_best_epsilon(num_trails: int, return_score: bool = False):
    print(f"Searching for best epsilon.")
    print(f"Trying {num_trails} different epsilons.")
    # Get random epsilon
    params = [get_random_epsilon() for _ in range(num_trails)]

    avg_scores = []
    X_train, y_train = get_dataset(DataSet.TRAIN_SET)

    for trail_id, epsilon in enumerate(params):
        print(
            f"\nTrying epsilon {trail_id + 1} out of {num_trails}.\nEpsilon is {epsilon:.3f}")
        model = DecisionTreeClassifier().use_alg(cost_sensitive_id3, extra_args={"epsilon": epsilon})
        avg_score = k_cross_validation(model, X_train, y_train, classification_rate)
        avg_scores.append(avg_score)
        print(f"Average validation score for this epsilon: {avg_score}\n")

    best_epsilon = params[int(np.argmin(avg_scores))]
    print(f"Best epsilon found: epsilon = {best_epsilon:.3f}")
    print(f"Average score for best params is: {np.min(avg_scores)}")
    if return_score:
        return best_epsilon, np.min(avg_scores)
    else:
        return best_epsilon


def main(args=None):
    def get_parser():
        _parser = argparse.ArgumentParser(
            'CostSensitiveID3',
            formatter_class=argparse.RawTextHelpFormatter,
            description=__doc__,
        )
        _parser.add_argument('-find', '--find', type=int, default=None,
                             help=f"Find optimal epsilon to use.\nUse `-find [NUM_PARAMS_TO_TRY]` to try "
                                  f"`NUM_PARAMS_TO_TRY`(int) different params.\n"
                                  f"Default is best found earlier by Almog: epsilon = {BEST_EPSILON}\n"
                                  f"Finding best epsilon is using randomly select `NUM_PARAMS_TO_TRY` times and find the one that "
                                  f"minimizes the average loss on the validation-set.\n"
                                  f"** NOTE ** : EPSILON_MIN, EPSILON_MAX can be found at constants.py and you can change them as well."
                             )
        return _parser

    parser = get_parser()
    args = parser.parse_args(args)

    # Find best parameters
    if args.find is not None:
        find_best_epsilon(num_trails=args.find)
    else:
        epsilon = BEST_EPSILON

        # Train model on entire train-set
        X_train, y_train = get_dataset(DataSet.TRAIN_SET)
        dt = DecisionTreeClassifier().use_alg(cost_sensitive_id3, extra_args={"epsilon": epsilon}).fit(X_train, y_train)

        # Evaluate model on test-set
        X_test, y_test = get_dataset(DataSet.TEST_SET)
        y_hat = dt.predict(X_test)
        print(ten_times_penalty(y_hat, y_test))


if __name__ == '__main__':
    main()
