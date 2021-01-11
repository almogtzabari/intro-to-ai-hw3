from DecisionTree import DecisionTreeClassifier
from utils import *

DecisionTreeNode = DecisionTreeClassifier.DecisionTreeNode


def select_feature_based_on_max_information_gain(X: np.ndarray, y: np.ndarray):
    def _feature_ig(X: np.ndarray, y: np.ndarray, feature, threshold):
        def _calc_entropy(y: np.ndarray):
            entropy = 0
            if len(y) == 0:
                return entropy
            _, counts = np.unique(y, return_counts=True)
            for count in counts:
                if count > 0:
                    p = count / len(y)
                    entropy -= p * np.log2(p)
            return entropy

        bigger_equal_indices = (X.T[feature] >= threshold)
        p_bigger_equal = bigger_equal_indices.sum() / len(X)
        smaller_indices = (X.T[feature] < threshold)
        p_smaller = smaller_indices.sum() / len(X)
        return _calc_entropy(y) - (p_bigger_equal * _calc_entropy(y[bigger_equal_indices])) - (
                    p_smaller * _calc_entropy(y[smaller_indices]))

    max_ig = float("-inf")
    max_feature = None
    max_feature_threshold = None
    for feature in range(X.shape[-1]):
        possible_values = np.unique(X.T[feature])
        thresholds = (possible_values[:-1] + possible_values[1:]) / 2
        for threshold in thresholds:
            ig = _feature_ig(X, y, feature, threshold)
            if ig >= max_ig:
                max_ig = ig
                max_feature = feature
                max_feature_threshold = threshold

    return max_feature, max_feature_threshold


def ID3(X: np.ndarray, y: np.ndarray):
    c = calc_most_common_value(y)
    return TDIDT(X, y, c, select_feature_based_on_max_information_gain)


def TDIDT(X, y, default, feature_select_fn: callable):
    if len(X) == 0:
        return DecisionTreeNode(None, None, [], default)  # Empty leaf. Use default classification

    c, c_count = calc_most_common_value(y, return_count=True)
    if len(y) == c_count or X.shape[-1] == 0:
        return DecisionTreeNode(None, None, [], c)

    # Select feature
    feature, threshold = feature_select_fn(X, y)  # Todo: Should we drop a feature that is continuous?

    # Separate dataset
    bigger_equal_indices = (X.T[feature] >= threshold)
    smaller_indices = (X.T[feature] < threshold)

    bigger_equal_sub_tree = TDIDT(X[bigger_equal_indices], y[bigger_equal_indices], c, feature_select_fn)
    smaller_sub_tree = TDIDT(X[smaller_indices], y[smaller_indices], c, feature_select_fn)
    return DecisionTreeNode(feature, threshold, [smaller_sub_tree, bigger_equal_sub_tree], c)


if __name__ == '__main__':
    # Train model
    X_train, y_train = get_dataset(data_set=DataSet.TRAIN_SET)
    dt = DecisionTreeClassifier().use_alg(ID3).fit(X_train, y_train)

    # Test model
    X_test, y_test = get_dataset(DataSet.TEST_SET)
    y_hat = dt.predict(X_test)

    # Print Results
    print(np.sum(y_hat == y_test) / len(y_test))
