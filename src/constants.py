import os
from enum import Enum


## Constants
SRC_DIR = os.path.dirname(__file__)
TRAIN_FILENAME = os.path.join(SRC_DIR, "train.csv")
TEST_FILENAME = os.path.join(SRC_DIR, "test.csv")
ID = 312433576
N_SPLITS = 5

# KNNForest constants
P_MIN = 0.3
P_MAX = 0.7
N_MIN = 5
N_MAX = 30
K_MIN = N_MIN
K_MAX = N_MAX
BEST_PARAMS = (16, 7, 0.49)


class DataSet(Enum):
    TRAIN_SET = 0
    TEST_SET = 1