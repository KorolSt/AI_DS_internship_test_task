# Animal Classification and Named Entity Recognition Pipeline

This project implements a pipeline that combines Named Entity Recognition (NER) and Image Classification to verify if a user's statement about an image is correct. The pipeline consists of two models:

1. **NER Model**: Extracts animal names from text. Uses BART model that was pretrained on labeled dataset
2. **Image Classification Model**: Classifies animals in images. Uses MobileNetV2 model that was pretrained on animals dataset 

The pipeline takes a text input (e.g., "There is a cow in the picture.") and an image as input and outputs a boolean value indicating whether the statement is correct.

 
---

## Requirements

To run this project, you need the following dependencies:
- Python 3.8 or higher
- Libraries listed in `requirements.txt`

Install the dependencies using:
```bash
    pip install -r requirements.txt
```

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
    ```
2. Train the models:
   -  Train the NER model:
    ```bash
    python train_ner.py --data_path task2/data/ner_labeled_dataset.jsonl
    ```
   -  Train the image classification model:
    ```bash
    python train_image_classifier.py --dataset_dir task2/data/animals_img_dataset/
    ```
  
# Usage 
Run the pipeline:
```bash
python src/pipeline.py --text "There is a cow in the picture." 
--image_path task2/data/animals_img_dataset/cat/1.jpeg 
--ner_model_dir task2/models/ner_model 
--image_model_path task2/models/image_classification_model.keras
```

# Datasets
- For NER model training was used synthetically created and manually labeled dataset
- For Image Classification Model Animals-10 dateset from Kaggle was used 

# Demo
Is available in `demo.ipynb`

# Exploratory Data Analysis (EDA)
Is available in `data_analysis.ipynb`

# Results
The models achieve the following performance:

- NER Model: Test loss of 9% on the test set.

- Image Classification Model: Accuracy of 84% on the test set.



