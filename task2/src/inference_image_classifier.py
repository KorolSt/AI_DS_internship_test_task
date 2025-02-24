"""
Script for predicting the class of an image using a pre-trained Keras model.

Usage Example:
    python predict_image_class.py --model "task2/models/image_classification_model.keras"
    --class_names "task2/data/class_names.json" --image_path "path/to/image.jpg" --target_size 224 224
"""

import argparse
import json
import logging
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress all logs except errors
tf.get_logger().setLevel(logging.ERROR)


def load_model(model_path: str):
    """
    Load a trained Keras model from the specified path.

    Args:
        model_path (str): Path to the trained model.

    Returns:
        tf.keras.Model: Loaded Keras model.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found at path: {model_path}")
    return tf.keras.models.load_model(model_path)


def load_classes(classes_names_path: str):
    """
    Load class names from a JSON file.

    Args:
        classes_names_path (str): Path to the JSON file containing class names.

    Returns:
        list: List of class names.

    Raises:
        FileNotFoundError: If the JSON file does not exist.
    """
    if not os.path.isfile(classes_names_path):
        raise FileNotFoundError(f"JSON file not found at path: {classes_names_path}")
    with open(classes_names_path, "r") as file:
        return json.load(file)


def preprocess_image(image_path: str, target_size: tuple):
    """
    Load and preprocess an image for prediction.

    Args:
        image_path (str): Path to the image file.
        target_size (tuple): Target size for resizing the image (height, width).

    Returns:
        np.ndarray: Preprocessed image array.

    Raises:
        FileNotFoundError: If the image file does not exist.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
    return img_array


def predict_image_class(image_path: str, model, class_names: list, target_size: tuple = (224, 224)):
    """
    Predict the class of an image using a trained model.

    Args:
        image_path (str): Path to the image file.
        model (tf.keras.Model): Trained Keras model.
        class_names (list): List of class names.
        target_size (tuple): Target size for resizing the image (height, width).

    Returns:
        tuple: Predicted class name and confidence score.
    """
    # Preprocess the image
    img_array = preprocess_image(image_path, target_size)

    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    prediction_index = np.argmax(predictions)
    predicted_class = class_names[prediction_index]
    confidence = np.max(predictions) * 100  # Confidence score as a percentage

    return predicted_class, confidence

def main():
    parser = argparse.ArgumentParser(description="Infer the class of an image using a trained model.")
    parser.add_argument("--model", type=str, default="task2/models/image_classification_model.keras",
                        help="Path to the trained Keras model.")
    parser.add_argument("--class_names", type=str, default="task2/data/class_names.json",
                        help="Path to the JSON file containing class names.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image file.")
    parser.add_argument("--target_size", type=int, nargs=2, default=[224, 224],
                        help="Target size for resizing the image (height, width).")

    args = parser.parse_args()

    # Load model and class names
    model = load_model(args.model)
    class_names = load_classes(args.class_names)

    # Predict image class
    predicted_class, confidence = predict_image_class(args.image_path, model, class_names, tuple(args.target_size))
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")


if __name__ == "__main__":
    main()