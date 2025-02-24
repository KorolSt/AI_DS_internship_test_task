import numpy as np

from task1.mnist_classifier_interface import MnistClassifierInterface
from task1.cnn_model import CNN
from task1.fnn_model import FNN
from task1.rf_model import RF


class MnistClassifier:
    """
    Wrapper class to select and use a specific MNIST classifier.
    """

    def __init__(self, algorithm: str):
        # Initialize the selected classifier based on the algorithm
        if algorithm == "cnn":
            self.classifier = CNN()
        elif algorithm == "nn":
            self.classifier = FNN()
        elif algorithm == "rf":
            self.classifier = RF()
        else:
            raise ValueError("Invalid algorithm. Choose 'cnn', 'nn', or 'rf'.")

    def train(self, x_train, y_train):
        self.classifier.train(x_train, y_train)

    def predict(self, x_test):
        return self.classifier.predict(x_test)
