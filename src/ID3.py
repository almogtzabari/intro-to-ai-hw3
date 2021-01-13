import argparse
from DecisionTree import DecisionTreeClassifier
from utils import *
from sklearn.model_selection import KFold
from typing import Sequence

DecisionTreeNode = DecisionTreeClassifier.DecisionTreeNode


def select_feature_based_on_max_information_gain(X: np.ndarray, y: np.ndarray):
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

    max_ig = float("-inf")
    max_feature = None
    max_feature_threshold = None
    for feature in range(X.shape[-1]):
        possible_values = np.unique(X.T[feature])
        thresholds = (possible_values[:-1] + possible_values[1:]) / 2
        for threshold in thresholds:
            ig = _feature_ig(X, y, feature, threshold)
            if ig == max_ig and feature == max_feature:
                # Only threshold is different. use previous threshold (see FAQ).
                continue
            if ig >= max_ig:
                max_ig = ig
                max_feature = feature
                max_feature_threshold = threshold

    return max_feature, max_feature_threshold


def ID3(X: np.ndarray, y: np.ndarray):
    c = calc_most_common_value(y)
    return TDIDT(X, y, c, select_feature_based_on_max_information_gain)


def ID3_with_early_pruning(X: np.ndarray, y: np.ndarray, M: int):
    c = calc_most_common_value(y)
    return TDIDT(X, y, c, select_feature_based_on_max_information_gain, M=M)


def TDIDT(X, y, default, feature_select_fn: callable, M=None):
    if len(X) == 0 or (M is not None and len(X) < M):
        return DecisionTreeNode(None, None, [], default)

    c, c_count = calc_most_common_value(y, return_count=True)
    if len(y) == c_count or X.shape[-1] == 0:
        return DecisionTreeNode(None, None, [], c)

    # Select feature
    feature, threshold = feature_select_fn(X, y)

    # Separate dataset
    bigger_equal_indices = (X.T[feature] >= threshold)
    smaller_indices = (X.T[feature] < threshold)

    bigger_equal_sub_tree = TDIDT(X[bigger_equal_indices], y[bigger_equal_indices], c, feature_select_fn, M)
    smaller_sub_tree = TDIDT(X[smaller_indices], y[smaller_indices], c, feature_select_fn, M)
    return DecisionTreeNode(feature, threshold, [smaller_sub_tree, bigger_equal_sub_tree], c)


def find_best_M(X_train: np.ndarray, y_train: np.ndarray, M_values: Sequence[int], evaluate_fn: callable,
                minimize: bool = False, return_score: bool = False):
    avg_score = []

    # Split indices
    train_indices = []
    val_indices = []
    for train_idx, val_idx in KFold(n_splits=N_SPLITS, shuffle=True, random_state=ID).split(X_train):
        train_indices.append(train_idx)
        val_indices.append(val_idx)

    # Run K-Fold-Cross-Validation
    for M in M_values:
        print(f"Running on M={M}")
        score = []
        for i in range(len(train_indices)):
            # Train model with value M
            train_idx = train_indices[i]
            curr_X_train = X_train[train_idx]
            curr_y_train = y_train[train_idx]
            dt = DecisionTreeClassifier().use_alg(ID3_with_early_pruning, extra_args={"M": M}).fit(curr_X_train,
                                                                                                   curr_y_train)

            # Validate
            val_idx = val_indices[i]
            curr_X_val = X_train[val_idx]
            curr_y_val = y_train[val_idx]
            curr_y_hat = dt.predict(curr_X_val)
            score.append(evaluate_fn(curr_y_hat, curr_y_val))
        avg_score.append(np.average(score))

    if minimize:
        best_M = M_values[int(np.argmin(np.array(avg_score)))]
    else:
        best_M = M_values[int(np.argmax(np.array(avg_score)))]
    if return_score:
        return best_M, avg_score
    else:
        return best_M


def experiment():
    M_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30]
    X_train, y_train = get_dataset(data_set=DataSet.TRAIN_SET)

    best_M, avg_accuracy = find_best_M(X_train, y_train, M_values, classification_rate, minimize=False, return_score=True)
    plot_graph(M_values, avg_accuracy, "Average Accuracy per M", "M", "Average Accuracy")

    # Train the model on the entire train-set
    dt = DecisionTreeClassifier().use_alg(ID3_with_early_pruning, extra_args={"M": best_M}).fit(X_train, y_train)

    # Predict and evaluate
    X_test, y_test = get_dataset(DataSet.TEST_SET)
    y_hat = dt.predict(X_test)
    print(classification_rate(y_hat, y_test))


def question1() -> None:
    """
    This function is used for question1.
    Running this function will fit a DecisionTreeClassifier with ID3 algorithm and print its accuracy on the test-set.
    """
    # Train model
    X_train, y_train = get_dataset(data_set=DataSet.TRAIN_SET)
    dt = DecisionTreeClassifier().use_alg(ID3).fit(X_train, y_train)

    # Test model
    X_test, y_test = get_dataset(DataSet.TEST_SET)
    y_hat = dt.predict(X_test)

    # Print Results
    print(classification_rate(y_hat, y_test))


def question3() -> None:
    """
    This function is used for question3.
    The function will find the best M value for early pruning (ID3 with M value) using K-Cross-Validation, and draw a
    graph of the average accuracy as a function of the M value used.
    Finally, this function will fit a DecisionTreeClassifier (with ID3 and the best M value found an) on the entire
    train-set and print its accuracy on the test-set.
    """
    experiment()


def question4():
    M_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30]
    X_train, y_train = get_dataset(data_set=DataSet.TRAIN_SET)

    # Find best M using new loss function
    best_M = find_best_M(X_train, y_train, M_values, ten_times_penalty, minimize=True)
    print(f"Best M is: {best_M}")

    # Train model on entire train-set using the best M found
    dt = DecisionTreeClassifier().use_alg(ID3_with_early_pruning, extra_args={"M": best_M}).fit(X_train, y_train)

    # Test model
    X_test, y_test = get_dataset(DataSet.TEST_SET)
    y_hat = dt.predict(X_test)

    # Print Results
    print(ten_times_penalty(y_hat, y_test))


def main(args=None):
    def get_parser():
        """
        Creates a new argument parser.
        """
        _parser = argparse.ArgumentParser('ID3')
        _parser.add_argument('-q', '--q', type=int, default=1, choices=[1, 3, 4],
                            help="Choose which question to run (Default: 1).")
        return _parser

    parser = get_parser()
    args = parser.parse_args(args)
    question_function_dict = {
        1: question1,
        3: question3,
        4: question4,
    }
    return question_function_dict[args.q]()


if __name__ == '__main__':
    main()
