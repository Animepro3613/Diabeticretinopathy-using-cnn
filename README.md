# APTOS 2019 Blindness Detection

## Overview
This project implements a convolutional neural network (CNN) for detecting diabetic retinopathy using the APTOS 2019 dataset. The model achieves an accuracy of **94.5%** on the test set. The train dataset consists of **3662 images**, which was preprocessed and seprated into train and test at random , the preprocessed images are included in this repository.

## Dataset
The images for this project can be downloaded from the official APTOS 2019 Blindness Detection competition page. Due to their large size, the original images were not uploaded to this repository.

- **Download Link**: [APTOS 2019 Blindness Detection Dataset](https://www.kaggle.com/c/aptos2019-blindness-detection/data)

## Model Performance
- **Accuracy**: 94.5%
- The model was trained and evaluated using the preprocessed images from the APTOS dataset.

## Explainability
To enhance the interpretability of the model's predictions, **Shapley values** are utilized. Given that the input data consists of images, Shapley values provide a robust method for explainability. The SHAP output is overlaid on the images, allowing for a visual understanding of the model's decision-making process.

## Requirements
To run this project, you will need the following libraries:
- TensorFlow
- NumPy
- Pandas
- OpenCV
- SHAP
- Matplotlib
- TQDM

You can install the required libraries using pip:
```bash
pip install tensorflow numpy pandas opencv-python shap matplotlib tqdm
