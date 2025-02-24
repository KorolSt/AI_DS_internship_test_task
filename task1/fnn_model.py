import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from task1.mnist_classifier_interface import MnistClassifierInterface


class FNN(MnistClassifierInterface):
    """
    Feed-Forward Neural Network (FNN) model for MNIST classification.
    """

    def __init__(self):
        self.model = Sequential(
            [
                Input(shape=(28, 28, 1)),
                Flatten(),
                Dense(128, activation="relu", name="L1"),
                Dense(64, activation="relu", name="L2"),
                Dense(10, activation="softmax", name="L3")
            ], name="fnn_model"
        )
        self.model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    def train(self, X_train, y_train):
        history = self.model.fit(X_train, y_train, epochs=10, batch_size=128, verbose=1)
        print(history.history.keys())

    def predict(self, X_test):
        return np.argmax(self.model.predict(X_test), axis=1)