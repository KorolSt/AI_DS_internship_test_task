"""
Train a Named Entity Recognition (NER) model using BERT.

Usage Example:
    python train_ner.py --data_path task2/data/ner_labeled_dataset.jsonl
    --output_dir task2/models/ner_model --epochs 4 --batch_size 32 --learning_rate 3e-5
"""

import argparse
import json

from datasets import Dataset
from transformers import (
    BertForTokenClassification,
    BertTokenizerFast,
    Trainer,
    TrainingArguments,
)

# Label mappings
LABEL2ID = {"O": 0, "B-ANIMAL": 1, "I-ANIMAL": 2, "B-NEGATED": 3, "I-NEGATED": 4}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

def convert_to_token_labels(example):
    """
    Convert raw text and labels into token-level labels.

    Args:
        example (dict): A dictionary containing "text" and "label" keys.

    Returns:
        dict: A dictionary with "tokens" and "labels" keys.
    """
    tokens = example["text"].split()
    labels = ["O"] * len(tokens)
    for start, end, entity_type in example["label"]:
        start_idx = 0
        for i, token in enumerate(tokens):
            end_idx = start_idx + len(token)
            if start < end_idx and end > start_idx:
                labels[i] = f"B-{entity_type}" if labels[i] == "O" else f"I-{entity_type}"
            start_idx = end_idx + 1
    example["tokens"] = tokens
    example["labels"] = [LABEL2ID[label] for label in labels]
    return example


def tokenize_and_align_labels(examples, tokenizer):
    """
    Tokenize text and align labels with tokenized inputs.

    Args:
        examples (dict): A dictionary containing "tokens" and "labels" keys.
        tokenizer: Pre-trained tokenizer.

    Returns:
        dict: Tokenized inputs with aligned labels.
    """
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True, padding="max_length", max_length=128
    )
    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = [label[word_idx] if word_idx is not None else -100 for word_idx in word_ids]
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def parse_args():
    parser = argparse.ArgumentParser(description="Train a NER model with BERT.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the JSONL dataset.")
    parser.add_argument("--output_dir", type=str, default="task2/models/ner_model", help="Directory to save the trained model.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load and process dataset
    with open(args.data_path, "r") as file:
        data = [convert_to_token_labels(json.loads(line)) for line in file]

    dataset = Dataset.from_dict({
        "tokens": [example["tokens"] for example in data],
        "labels": [example["labels"] for example in data]
    })

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    tokenized_dataset = dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)
    split_dataset = tokenized_dataset.train_test_split(test_size=0.2)

    model = BertForTokenClassification.from_pretrained(
        "bert-base-uncased", num_labels=len(LABEL2ID), id2label=ID2LABEL, label2id=LABEL2ID
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_dir=f"{args.output_dir}/logs",
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        tokenizer=tokenizer,
    )

    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()