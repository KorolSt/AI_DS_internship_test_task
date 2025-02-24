import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from task1.mnist_classifier_interface import MnistClassifierInterface


class CNN(MnistClassifierInterface):
    """
    CNN model for MNIST classification.
    """

    def __init__(self):
        self.model = Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train(self, X_train, y_train):
        history = self.model.fit(X_train, y_train, epochs=10, batch_size=128)
        print(history.history.keys())

    def predict(self, X_test):
        return np.argmax(self.model.predict(X_test), axis=1)


