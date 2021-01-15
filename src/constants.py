import os
from enum import IntEnum


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
# BEST_KNN_FOREST_PARAMS = (16, 7, 0.49)
BEST_KNN_FOREST_PARAMS = (8, 5, 0.51)

TEMPERATURE_MIN = 1
TEMPERATURE_MAX = 10
BEST_IMPROVED_KNN_FOREST_PARAMS = (5, 3, 0.51, 1)


class DataSet(IntEnum):
    TRAIN_SET = 0
    TEST_SET = 1


class Classification(IntEnum):
    HEALTHY = -1  # This is B
    SICK = 1  # This is M
