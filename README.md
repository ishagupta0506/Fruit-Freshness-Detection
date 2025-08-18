# Fruit Freshness Detection

This repository contains an image-processing and machine learning pipeline for detecting the freshness of fruits using computer vision techniques. The implementation is based on Python and common data science libraries. A simple yet effective image classification notebook that distinguishes fresh fruits from rotten ones across three categories: apples, bananas, and oranges.

## Project Overview

The goal of this project is to classify fruits as **fresh** or **rotten** using a dataset of fruit images.
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
├── archive (2)/          # Dataset 
├── fruit freshness.ipynb # Main Jupyter Notebook with pipeline
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## Dataset

The project uses the **[Fruits Fresh and Rotten for Classification](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification)** dataset from Kaggle. The dataset contains **13,599 images** of fruit, organized into training and testing splits with six classes:

```
dataset/
├── train/
│   ├── freshapples/     # ~1693 images
│   ├── freshbanana/     # ~1581 images
│   ├── freshoranges/    # ~1466 images
│   ├── rottenapples/    # ~2342 images
│   ├── rottenbanana/    # ~2224 images
│   └── rottenoranges/   # ~1595 images
└── test/
    ├── freshapples/     # ~395 images
    ├── freshbanana/     # ~381 images
    ├── freshoranges/    # ~388 images
    ├── rottenapples/    # ~601 images
    ├── rottenbanana/    # ~530 images
    └── rottenoranges/   # ~403 images
```

* **Total images**: \~13.6K
* **Classes**: 6 (fresh vs. rotten for apples, bananas, oranges)
  ([GitHub][1])

---
[1]: https://github.com/Bangkit-JKT2-D/fruits-fresh-rotten-classification?utm_source=chatgpt.com "Bangkit-JKT2-D/fruits-fresh-rotten-classification - GitHub"

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
### Dependencies for Seamless Use of the Notebook

To ensure smooth execution of the Jupyter notebook, please install the following dependencies:

```bash
pip install jupyter>=1.0.0 notebook>=6.4.0
```

It is also recommended to install commonly used libraries for deep learning and data processing (if not already available):

```bash
pip install numpy pandas matplotlib seaborn tensorflow keras scikit-learn
```

---

### Recommended Enhancements

To further improve the model and project outcomes, consider the following enhancements:

* **Validation Split:** Create a validation set to monitor model performance during training.
* **CNN Implementation:** Train a custom Convolutional Neural Network for better accuracy compared to traditional ML models.
* **Transfer Learning:** Leverage pre-trained models (e.g., ResNet50, EfficientNet, MobileNet) for faster convergence and higher accuracy.
* **Data Augmentation:** Apply techniques like rotation, flipping, zooming, and brightness adjustments to increase dataset variability and reduce overfitting.
* **Hyperparameter Tuning:** Experiment with batch size, learning rate, and number of epochs for optimal results.
* **Explainability Tools:** Use Grad-CAM or SHAP to visualize what parts of the fruit images the model focuses on when making predictions.

---

## Evaluation Metrics

The model can be evaluated using:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-score**
* **Confusion Matrix**

---


## License

This project is licensed under the MIT License.

