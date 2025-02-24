# TASK 1


This project implements three machine learning models for classifying handwritten digits from the MNIST dataset:
1. **Random Forest (RF)**
2. **Feed-Forward Neural Network (FNN)**
3. **Convolutional Neural Network (CNN)**

Each model is implemented as a separate class and follows a common interface (`MnistClassifierInterface`). The `MnistClassifier` class acts as a wrapper to select and use the desired model.

## Solution Explanation

### Models
1. **Random Forest (RF)**:
   - A traditional machine learning model using an ensemble of decision trees.
   - Input images are flattened into 1D vectors before training and prediction.

2. **Feed-Forward Neural Network (FNN)**:
   - A simple neural network with two hidden layers and ReLU activation.
   - Input images are flattened into 1D vectors before being fed into the network.

3. **Convolutional Neural Network (CNN)**:
   - A deep learning model designed for image data.
   - Uses convolutional and pooling layers to extract features, followed by fully connected layers for classification.

### Interface
- All models implement the `MnistClassifierInterface`, which defines two methods:
  - `train(X_train, y_train)`: Trains the model on the provided data.
  - `predict(X_test)`: Predicts class labels for the provided test data.

### Wrapper Class
- The `MnistClassifier` class allows selecting a model (`cnn`, `nn`, or `rf`) and provides a unified interface for training and prediction.

---

## Setup Instructions


### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name

2. Install the required dependencies:
    ```bash
   pip install -r requirements.txt
   
3. Run the Jupyter Notebook `demo.ipynb` file to train and evaluate the models.