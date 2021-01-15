import pandas as pd
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from constants import *


def calc_most_common_value(array: np.ndarray, return_count=False):
    if not isinstance(array, np.ndarray):
        raise Exception(f"Given `array` is not a numpy array!")
    vals, counts = np.unique(array, return_counts=True)
    if len(vals) == 0:
        return None
    common_value, common_occurrences = vals[counts.argmax()], counts.max()
    return common_value, common_occurrences if return_count else common_value


def get_dataset(data_set: DataSet = DataSet.TRAIN_SET) -> Tuple[np.ndarray, np.ndarray]:
    def _read_train_set() -> pd.DataFrame:
        """ Returns a pandas object with the train data. """
        return pd.read_csv(TRAIN_FILENAME)

    def _read_test_set() -> pd.DataFrame:
        """ Returns a pandas object with the test data. """
        return pd.read_csv(TEST_FILENAME)

    df = _read_train_set() if data_set == DataSet.TRAIN_SET else _read_test_set()
    X = df.drop(columns="diagnosis").to_numpy()
    _y = df.diagnosis.to_numpy()

    # Encode y
    y = np.zeros_like(_y)
    y[_y == "M"] = int(Classification.SICK)
    y[_y == "B"] = int(Classification.HEALTHY)
    return X, y


def get_and_split_dataset(data_set: DataSet = DataSet.TRAIN_SET, ratio: float = 0.2) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = get_dataset(data_set)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio, shuffle=True, random_state=ID)
    return X_train, y_train, X_test, y_test


def plot_graph(x, y, title=None, x_label=None, y_label=None):
    plt.plot(x, y)
    if title is not None:
        plt.title(title)
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    plt.show()


def classification_rate(y_hat: np.ndarray, y: np.ndarray):
    if len(y_hat) != len(y):
        raise Exception(f"Dimension error!")
    if len(y_hat) == 0:
        return 1
    return np.average(y_hat == y)


def ten_times_penalty(y_hat: np.ndarray, y: np.ndarray):
    if len(y_hat) != len(y):
        raise Exception(f"Dimension error!")
    if len(y_hat) == 0:
        return 0

    fp = (y_hat == int(Classification.SICK)) * (y == int(Classification.HEALTHY))  # Healthy people accidentally classified as sick (positive).
    fn = (y_hat == int(Classification.HEALTHY)) * (y == int(Classification.SICK))  # Sick people accidentally classified as healthy.
    return np.average(0.1 * fp + fn)


def get_random_params_for_knn_forest():
    N = np.random.randint(N_MIN, N_MAX)
    K = np.random.randint(K_MIN, N) if N > K_MIN else N
    p = np.random.uniform(P_MIN, P_MAX)
    return N, K, p


def get_random_params_for_improved_knn_forest():
    N, K, p = get_random_params_for_knn_forest()
    T = np.random.uniform(TEMPERATURE_MIN, TEMPERATURE_MAX)
    return N, K, p, T


def k_cross_validation(model, X: np.ndarray, y: np.ndarray, evaluate_fn: callable = classification_rate,
                       k: int = N_SPLITS) -> float:
    scores = []
    # Split indices
    train_indices = []
    val_indices = []
    for train_idx, val_idx in KFold(n_splits=k, shuffle=True, random_state=ID).split(X):
        train_indices.append(train_idx)
        val_indices.append(val_idx)

    for i, (train_idx, val_idx) in enumerate(zip(train_indices, val_indices)):
        model.fit(X[train_idx], y[train_idx])
        y_hat = model.predict(X[val_idx])
        scores.append(evaluate_fn(y_hat, y[val_idx]))
        print(f"Validation {i + 1} out of {len(train_indices)}, score: {scores[-1]}")

    return np.average(scores)


def std_normalization(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (X - mean) / std