"""
Train an image classification model using MobileNetV2.

Usage Example:
    python train_image_classifier.py --dataset_dir task2/data/animals_img/dataset/ --image_size 224 224 --batch_size 128
    --epochs 1 --model_save_path "task2/models/image_classification_model22.keras"
    --class_names_save_path "task2/models/class_names.json"
"""

import argparse
import json

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential


def load_dataset(dataset_path, image_size, batch_size):
    """
    Load and split the dataset into training and validation sets.

    Args:
        dataset_path (str): Path to the dataset directory.
        image_size (tuple): Target image size (height, width).
        batch_size (int): Batch size for training.

    Returns:
        tuple: Training dataset, validation dataset, and class names.
    """
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_path,
        labels="inferred",
        label_mode="int",
        image_size=image_size,
        batch_size=batch_size,
    )
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)
    print("Classes:", dataset.class_names)
    return train_dataset, val_dataset, dataset.class_names


def build_model(input_shape, num_classes):
    """
    Build a MobileNetV2-based image classification model with data augmentation.

    Args:
        input_shape (tuple): Input shape of the images (height, width, channels).
        num_classes (int): Number of output classes.

    Returns:
        tf.keras.Model: Compiled Keras model.
    """
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet", alpha=0.35)

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomBrightness(0.2),
        layers.RandomWidth(0.2),
        layers.RandomHeight(0.2),
    ])

    model = Sequential([
        data_augmentation,
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.Dropout(0.7),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_model(train_dataset, val_dataset, class_names, input_shape, epochs, model_save_path, class_names_save_path):
    """
    Train the image classification model and save it to disk.

    Args:
        train_dataset (tf.data.Dataset): Training dataset.
        val_dataset (tf.data.Dataset): Validation dataset.
        class_names (list): List of class names.
        input_shape (tuple): Input shape of the images (height, width, channels).
        epochs (int): Number of training epochs.
        model_save_path (str): Path to save the trained model.
        class_names_save_path (str): Path to save the class names.
    """
    model = build_model(input_shape, len(class_names))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=2),
    ]

    history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, callbacks=callbacks)
    test_loss, test_accuracy = model.evaluate(val_dataset)
    print(f"Validation Accuracy: {test_accuracy:.2f}")

    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    with open(class_names_save_path, "w") as file:
        json.dump(class_names, file)
    print(f"Class names saved to {class_names_save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train an image classification model.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224],
                        help="Target image size (height, width).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--model_save_path", type=str,
                        default="task2/models/image_classification_model.keras",
                        help="Path to save the trained model.")
    parser.add_argument("--class_names_save_path", type=str, default="task2/models/class_names.json",
                        help="Path to save the class names.")
    args = parser.parse_args()

    # Load dataset
    train_dataset, val_dataset, class_names = load_dataset(args.dataset_dir, tuple(args.image_size), args.batch_size)

    # Train model
    input_shape = (args.image_size[0], args.image_size[1], 3)
    train_model(train_dataset, val_dataset, class_names, input_shape, args.epochs, args.model_save_path,
                args.class_names_save_path)


if __name__ == "__main__":
    main()