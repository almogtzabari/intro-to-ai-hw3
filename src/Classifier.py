from abc import ABC


class Classifier(ABC):
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass
