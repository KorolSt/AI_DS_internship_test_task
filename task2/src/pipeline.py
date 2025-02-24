"""
Pipeline to verify if the user's statement about an image is correct.

Usage Example:
    python pipeline.py --text "There is a cow in the picture." --image_path "path/to/image.jpg" --ner_model_dir "task2/src/ner_model" --image_model_path "task2/models/image_classification_model.keras" --class_names_path "task2/models/class_names.json" --confidence_threshold 0.5 --target_size 224 224
"""

import argparse
import logging
import os

import tensorflow as tf
from transformers import logging as transformers_logging

from inference_ner import load_model_and_tokenizer, predict_animals
from inference_image_classifier import load_model, load_classes, predict_image_class

# Suppress Hugging Face logs
transformers_logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress all logs except errors
tf.get_logger().setLevel(logging.ERROR)


def pipeline(text, image_path, ner_model_dir, image_model_path, class_names_path, confidence_threshold=0.5,
             target_size=(224, 224)):
    """
    Pipeline to verify if the user's statement about the image is correct.

    Args:
        text (str): User's text input (e.g., "There is a cow in the picture.").
        image_path (str): Path to the image file.
        ner_model_dir (str): Directory containing the trained NER model.
        image_model_path (str): Path to the trained image classification model.
        class_names_path (str): Path to the JSON file containing class names.
        confidence_threshold (float): Confidence threshold for accepting NER predictions.
        target_size (tuple): Target size for resizing the image (height, width).

    Returns:
        bool: True if the user's statement is correct, False otherwise.
    """
    # Step 1: Extract animal names from text using NER model
    tokenizer, ner_model = load_model_and_tokenizer(ner_model_dir)
    animals_from_text = predict_animals(text, tokenizer, ner_model, confidence_threshold)
    logging.info(f"Extracted animals from text: {animals_from_text}")

    # Step 2: Predict animal in the image using the image classification model
    image_model = load_model(image_model_path)
    class_names = load_classes(class_names_path)
    predicted_class, _ = predict_image_class(image_path, image_model, class_names, target_size)
    logging.info(f"Predicted animal in image: {predicted_class}")

    # Step 3: Compare the results
    if predicted_class.lower() in [animal.lower() for animal in animals_from_text]:
        return True
    return False


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Pipeline to verify user's statement about an image.")
    parser.add_argument("--text", type=str, required=True, help="User's text input.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image file.")
    parser.add_argument("--ner_model_dir", type=str, default="task2/models/ner_model",
                        help="Directory containing the trained NER model.")
    parser.add_argument("--image_model_path", type=str,
                        default="task2/models/image_classification_model_84_accuracy.keras",
                        help="Path to the trained image classification model.")
    parser.add_argument("--class_names_path", type=str, default="task2/data/class_names.json",
                        help="Path to the JSON file containing class names.")
    parser.add_argument("--confidence_threshold", type=float, default=0.5,
                        help="Confidence threshold for NER predictions.")
    parser.add_argument("--target_size", type=int, nargs=2, default=[224, 224],
                        help="Target size for resizing the image (height, width).")
    return parser.parse_args()


def main():
    args = parse_args()

    # Run the pipeline
    result = pipeline(
        args.text,
        args.image_path,
        args.ner_model_dir,
        args.image_model_path,
        args.class_names_path,
        args.confidence_threshold,
        tuple(args.target_size)
    )
    print(f"Result: {result}")


if __name__ == "__main__":
    main()