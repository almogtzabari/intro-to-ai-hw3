import pandas as pd
import numpy as np
import os
from enum import Enum
from typing import Tuple
import matplotlib.pyplot as plt

SRC_DIR = os.path.dirname(__file__)
TRAIN_FILENAME = os.path.join(SRC_DIR, "train.csv")
TEST_FILENAME = os.path.join(SRC_DIR, "test.csv")


class DataSet(Enum):
    TRAIN_SET = 0
    TEST_SET = 1


def _read_train_set() -> pd.DataFrame:
    """ Returns a pandas object with the train data. """
    return pd.read_csv(TRAIN_FILENAME)


def _read_test_set() -> pd.DataFrame:
    """ Returns a pandas object with the test data. """
    return pd.read_csv(TEST_FILENAME)


def calc_most_common_value(array: np.ndarray, return_count=False):
    if not isinstance(array, np.ndarray):
        raise Exception(f"Given `array` is not a numpy array!")
    vals, counts = np.unique(array, return_counts=True)
    if len(vals) == 0:
        return None
    return vals[counts.argmax()], counts.max() if return_count else vals[counts.argmax()]


def get_dataset(data_set: DataSet = DataSet.TRAIN_SET) -> Tuple[np.ndarray, np.ndarray]:
    df = _read_train_set() if data_set == DataSet.TRAIN_SET else _read_test_set()
    y = df.diagnosis.to_numpy()
    X = df.drop(columns="diagnosis").to_numpy()
    return X, y


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
    return np.average(y_hat == y)


def ten_times_penalty(y_hat: np.ndarray, y: np.ndarray):
    fp = (y_hat == "M") * (y == "B")  # Healthy people accidentally classified as sick (positive).
    fn = (y_hat == "B") * (y == "M")  # Sick people accidentally classified as healthy.
    return np.average(0.1 * fp + fn)
