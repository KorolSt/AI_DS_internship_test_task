from task1.mnist_classifier_interface import MnistClassifierInterface
from sklearn.ensemble import RandomForestClassifier


class RF(MnistClassifierInterface):
    """
    Random Forest (RF) model for MNIST classification.
    """

    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)

    def train(self, x_train, y_train):
        # Reshape data for Random Forest (flatten 28x28 images to 784 features)
        x_train = x_train.reshape(x_train.shape[0], -1)
        print("Random Forest training...")
        self.model.fit(x_train, y_train)
        print("Random Forest training completed.")

    def predict(self, x_test):
        # Reshape data and predict class labels
        x_test = x_test.reshape(x_test.shape[0], -1)
        return self.model.predict(x_test)