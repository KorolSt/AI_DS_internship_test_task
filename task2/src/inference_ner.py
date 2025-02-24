"""
Script for predicting animal names from text using a pre-trained Named Entity Recognition (NER) model.

Usage Example:
    python predict_animals.py --text "There is a cow in the picture." --model_dir "task2/models/ner_model"
    --confidence_threshold 0.5
"""

import argparse

import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForTokenClassification


def load_model_and_tokenizer(model_dir):
    """
    Load the tokenizer and model from the specified directory.

    Args:
        model_dir (str): Directory containing the pre-trained model and tokenizer.

    Returns:
        tuple: Tokenizer and model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = TFAutoModelForTokenClassification.from_pretrained(model_dir)
    return tokenizer, model


def predict_animals(text, tokenizer, model, confidence_threshold=0.5):
    """
    Predict animal names from the input text.

    Args:
        text (str): Input text.
        tokenizer: Pre-trained tokenizer.
        model: Pre-trained model.
        confidence_threshold (float): Confidence threshold for accepting predictions.

    Returns:
        list: List of extracted animal names.
    """
    # Label mappings
    LABEL_LIST = ["O", "B-ANIMAL", "I-ANIMAL", "B-NEGATED", "I-NEGATED"]
    ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}

    # Tokenize input text
    inputs = tokenizer(text, return_tensors="tf")
    outputs = model(**inputs).logits
    probabilities = tf.nn.softmax(outputs, axis=-1)[0].numpy()
    predictions = tf.argmax(outputs, axis=-1)[0].numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # Extract animal names and check for negation
    animals = []
    current_animal = ""
    current_score_sum = 0.0
    current_token_count = 0
    is_negated = False  # Flag to track if negation is present

    for token, pred, token_probs in zip(tokens, predictions, probabilities):
        label = ID2LABEL[pred]
        score = token_probs[pred]

        # Check for negation
        if label in ["B-NEGATED", "I-NEGATED"]:
            is_negated = True

        # Extract animal names
        if label == "B-ANIMAL":
            if current_animal and (current_score_sum / current_token_count) > confidence_threshold:
                animals.append(current_animal.strip())
            current_animal = token.replace("##", "") + " "
            current_score_sum = score
            current_token_count = 1
        elif label == "I-ANIMAL" and current_animal:
            current_animal += token.replace("##", "") + " "
            current_score_sum += score
            current_token_count += 1
        else:
            if current_animal and (current_score_sum / current_token_count) > confidence_threshold:
                animals.append(current_animal.strip())
            current_animal = ""
            current_score_sum = 0.0
            current_token_count = 0

    # Check for the last entity
    if current_animal and (current_score_sum / current_token_count) > confidence_threshold:
        animals.append(current_animal.strip())

    # If negation is present, return an empty list
    if is_negated:
        return []

    return animals


def main():
    parser = argparse.ArgumentParser(description="Infer animal mentions from text.")
    parser.add_argument("--text", type=str, required=True, help="Input text to analyze.")
    parser.add_argument("--model_dir", type=str, default="task2/models/ner_model",
                        help="Directory containing the trained model.")
    parser.add_argument("--confidence_threshold", type=float, default=0.5,
                        help="Confidence threshold for accepting predictions.")
    args = parser.parse_args()

    # Load model and tokenizer
    tokenizer, model = load_model_and_tokenizer(args.model_dir)

    # Predict animal names
    detected_animals = predict_animals(args.text, tokenizer, model, args.confidence_threshold)
    print(detected_animals)


if __name__ == "__main__":
    main()