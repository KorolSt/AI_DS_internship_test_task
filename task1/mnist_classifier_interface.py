from abc import ABC, abstractmethod


class MnistClassifierInterface(ABC):
    """
    Interface for MNIST classifiers. All classifiers must implement `train` and `predict`.
    """

    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass