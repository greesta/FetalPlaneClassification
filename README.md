# Fetal Plane Classification with EfficientNetV2-S and Grad-CAM

Deep learning-based classification of fetal ultrasound images into six fetal plane categories using **EfficientNetV2-S**, with **Grad-CAM** for visual interpretation of model predictions.

## Overview

This project implements a deep learning pipeline for classifying fetal ultrasound images into six major plane categories. The model uses **EfficientNetV2-S** with ImageNet pretrained weights as the backbone and is further fine-tuned for fetal plane classification.

To improve the interpretability of the model, **Gradient-weighted Class Activation Mapping (Grad-CAM)** is applied to visualize the regions of an ultrasound image that contribute to the model's prediction.

### Classification Classes

The model classifies ultrasound images into six categories:

1. **Fetal Abdomen**
2. **Fetal Brain**
3. **Fetal Femur**
4. **Fetal Thorax**
5. **Maternal Cervix**
6. **Other**

The Fetal Brain category includes subcategories such as Trans-thalamic, Trans-cerebellum, and Trans-ventricular planes.

## Dataset

The project uses the publicly available **Maternal-Fetal Plane Ultrasound Dataset** from ZENODO.

The dataset contains **12,400 ultrasound images** with associated metadata, including:

* `Image_name`
* `Patient_num`
* `Plane`
* `Brain_plane`
* `Operator`
* `US_Machine`
* `Train`

The dataset contains six `Plane` classes used as the classification target.

> The dataset itself is not included in this repository.

## Project Pipeline

The overall workflow consists of:

```text
Dataset
   ↓
Data Assessment
   ↓
Exploratory Data Analysis
   ↓
Stratified Train/Validation/Test Split
   ↓
Label Encoding
   ↓
Image Preprocessing
   ↓
Data Augmentation
   ↓
EfficientNetV2-S
   ↓
Initial Training
   ↓
Fine-tuning
   ↓
Model Evaluation
   ↓
Grad-CAM Visualization
```

## Data Preparation

The dataset is divided using a stratified split to maintain class distribution:

| Dataset    | Number of Images |
| ---------- | ---------------: |
| Training   |            8,679 |
| Validation |            1,861 |
| Testing    |            1,860 |
| **Total**  |       **12,400** |

The images are resized to:

```text
224 × 224 × 3
```

### Preprocessing

The preprocessing pipeline includes:

* Label encoding
* Dataset shuffling
* Image decoding
* Image resizing
* Tensor shape definition
* Batching
* Prefetching
* EfficientNetV2 preprocessing
* One-hot label encoding

### Data Augmentation

Augmentation is applied only to the training data:

* Random horizontal flip
* Random rotation: `0.05`
* Random zoom: `0.08`

## Model Architecture

The model uses **EfficientNetV2-S** pretrained on ImageNet as the feature extraction backbone.

The classification head consists of:

```text
EfficientNetV2-S
      ↓
Global Average Pooling
      ↓
Dense (256, Swish)
      ↓
Dropout (0.4)
      ↓
Dense (6, Softmax)
```

### Model Configuration

| Parameter          | Configuration            |
| ------------------ | ------------------------ |
| Backbone           | EfficientNetV2-S         |
| Pretrained weights | ImageNet                 |
| Input size         | 224 × 224 × 3            |
| Number of classes  | 6                        |
| Dense units        | 256                      |
| Activation         | Swish                    |
| Dropout            | 0.4                      |
| Output activation  | Softmax                  |
| Loss               | Categorical Crossentropy |
| Label smoothing    | 0.1                      |

## Training

### Phase 1: Initial Training

During the initial training phase, the EfficientNetV2-S backbone is frozen and only the classification head is trained.

Configuration:

```text
Optimizer: Adam
Learning rate: 1e-3
Epochs: 50
Label smoothing: 0.1
```

This stage allows the newly added classification layers to learn the characteristics of the six fetal plane classes before fine-tuning the pretrained backbone.

### Phase 2: Fine-tuning

Fine-tuning is performed by progressively unfreezing the EfficientNetV2-S backbone.

The configuration used in the notebook includes:

```text
Optimizer: AdamW
Learning rate: 1e-6
Weight decay: 1e-4
Label smoothing: 0.1
Maximum epochs: 50
```

Batch Normalization layers are kept frozen during fine-tuning.

Two callbacks are used:

* **EarlyStopping**

  * Monitors validation accuracy
  * Patience: 5
  * Restores best weights

* **ReduceLROnPlateau**

  * Monitors validation loss
  * Factor: 0.3
  * Patience: 3
  * Minimum learning rate: 1e-6

## Evaluation

The model is evaluated on the held-out test set using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

### Test Performance

The final fine-tuned model achieved:

| Metric             |     Result |
| ------------------ | ---------: |
| Test Accuracy      | **95.86%** |
| Macro Precision    |   **0.95** |
| Macro Recall       |   **0.95** |
| Macro F1-score     |   **0.95** |
| Weighted Precision |   **0.96** |
| Weighted Recall    |   **0.96** |
| Weighted F1-score  |   **0.96** |
| Test Samples       |  **1,860** |

### Classification Report

| Class           | Precision | Recall | F1-score | Support |
| --------------- | --------: | -----: | -------: | ------: |
| Fetal Abdomen   |      0.90 |   0.90 |     0.90 |     106 |
| Fetal Brain     |      0.98 |   1.00 |     0.99 |     464 |
| Fetal Femur     |      0.92 |   0.90 |     0.91 |     156 |
| Fetal Thorax    |      0.95 |   0.96 |     0.95 |     258 |
| Maternal Cervix |      1.00 |   1.00 |     1.00 |     244 |
| Other           |      0.95 |   0.94 |     0.94 |     632 |

The confusion matrix is generated from the test predictions to analyze correct classifications and misclassifications between the six classes.

## Grad-CAM

**Grad-CAM** is implemented to provide a visual explanation of the model's predictions.

The final convolutional layer used for Grad-CAM is:

```text
top_conv
```

The Grad-CAM pipeline:

```text
Input Ultrasound Image
        ↓
Resize to 224 × 224
        ↓
EfficientNetV2-S
        ↓
Prediction
        ↓
Gradient Calculation
        ↓
Grad-CAM Heatmap
        ↓
Heatmap Overlay
```

The visualization displays:

1. Original ultrasound image
2. Grad-CAM heatmap
3. Heatmap overlay on the original image

The heatmap is resized to the original image dimensions and overlaid onto the ultrasound image to highlight regions associated with the predicted class.

## Repository Structure

A recommended repository structure is:

```text
.
├── README.md
├── notebooks/
│   └── Fetal_Plane_Classification_EfficientNetV2_S_and_GradCAM.ipynb
├── models/
│   └── fetal_plane_model.keras
└── results/
    ├── confusion_matrix.png
    └── gradcam/
```

The dataset and large model files may be excluded from the repository depending on GitHub file-size limitations and dataset licensing.

## Requirements

The notebook was developed using Python and TensorFlow/Keras.

Main libraries:

```text
TensorFlow 2.20.0
NumPy
Pandas
Matplotlib
OpenCV
Scikit-learn
Seaborn
```

Install the required packages with:

```bash
pip install tensorflow numpy pandas matplotlib opencv-python scikit-learn seaborn
```

## Usage

### 1. Prepare the Dataset

Place the dataset in the following structure:

```text
data/
├── Images/
│   ├── Patient00001_....png
│   ├── Patient00002_....png
│   └── ...
└── FETAL_PLANES_DB_data.csv
```

The notebook automatically constructs image paths based on the `Image_name` column.

### 2. Run the Notebook

Open:

```text
notebooks/Fetal_Plane_Classification_EfficientNetV2_S_and_GradCAM.ipynb
```

The notebook can be executed using Google Colab or another compatible Jupyter environment.

For Google Colab, the original notebook uses Google Drive to access the dataset.

### 3. Train the Model

Run the notebook sequentially to:

1. Load the dataset
2. Assess the data
3. Perform EDA
4. Split the dataset
5. Encode labels
6. Prepare the TensorFlow datasets
7. Build EfficientNetV2-S
8. Train the classification head
9. Fine-tune the model
10. Evaluate the model
11. Save the trained model
12. Generate Grad-CAM visualizations

## Saved Model

The trained model is saved in Keras format:

```text
fetal_plane_model.keras
```

The notebook also saves a fine-tuned model to Google Drive for later use.

## Reproducibility

The train/validation/test split uses stratification with:

```text
random_state = 42
```

This helps maintain a similar class distribution across the training, validation, and test sets.

## Limitations

This project is intended as a research and experimental implementation of fetal plane classification.

The model should **not be used as a standalone clinical diagnostic system**. Model predictions and Grad-CAM visualizations require appropriate clinical interpretation and validation before any potential clinical application.

## Citation

If you use this implementation or build upon this work, please cite the original dataset and acknowledge this repository.

### Dataset

Maternal-Fetal Plane Ultrasound Dataset, available through ZENODO.

### Model

EfficientNetV2-S is used as the pretrained backbone for image classification.

### Explainability

Grad-CAM is used to visualize image regions contributing to the model prediction.

## Author

**Greestaviola Allodya Darmawan**

Biomedical Engineering
Universitas Airlangga
