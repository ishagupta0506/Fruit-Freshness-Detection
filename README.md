# Fruit Freshness Detection

This repository contains an image-processing and machine learning pipeline for detecting the freshness of fruits using computer vision techniques. The implementation is based on Python and common data science libraries, and is intended for educational and experimental purposes.

## Project Overview

The goal of this project is to classify fruits as **fresh** or **stale** using a dataset of fruit images.
It uses image preprocessing, feature extraction, and machine learning models for prediction.

### Key Features

* **Image Preprocessing**: Resizing, color conversion, and normalization
* **Feature Extraction**: Texture, color, and shape features from images
* **Machine Learning Models**: Classification models trained on extracted features
* **Visualization**: Display of results and data insights

## Project Structure

```
fruit-freshness/
├── sample_images/        # Example fruit images for testing
├── archive (2)/          # Dataset (zipped or raw)
├── fruit freshness.ipynb # Main Jupyter Notebook with pipeline
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/ishagupta0506/fruit-freshness.git
cd fruit-freshness
```

### 2. Install dependencies

Make sure you have Python 3.8+ installed, then run:

```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset

Place your dataset inside the `archive (2)/` directory or update the paths in the notebook.

**Supported image formats**: `.jpg`, `.png`

### 4. Run the pipeline

Open the Jupyter Notebook:

```bash
jupyter notebook "fruit freshness.ipynb"
```

Run all cells to preprocess data, train the model, and view results.

---

## Usage

* **Training**: Done directly in the notebook
* **Prediction**: Upload or select an image in the notebook to get freshness prediction
* **Visualization**: Includes matplotlib plots for dataset and results

---

## Requirements

The dependencies are listed in `requirements.txt`:

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
opencv-python>=4.6.0
scikit-image>=0.19.0
scikit-learn>=1.1.0
```

---

## Evaluation Metrics

The model can be evaluated using:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-score**
* **Confusion Matrix**

---

## Dataset

You can use any fruit image dataset — for example, publicly available fruit freshness datasets from Kaggle.
The dataset should be organized into separate folders for each class (e.g., `fresh/` and `stale/`).

---

## License

This project is licensed under the MIT License.

