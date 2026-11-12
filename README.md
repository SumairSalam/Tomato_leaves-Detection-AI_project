# 🍅 Tomato Leaf Disease Classification Using AI

A deep learning project that classifies tomato leaf images into **10 categories**, including healthy leaves, plant diseases, and spider mite damage.

The project uses **MobileNet transfer learning** with TensorFlow/Keras and is implemented in a Jupyter Notebook.

## Overview

This project explores how computer vision can help identify tomato leaf conditions from images.

The notebook covers:

- Dataset downloading and preparation
- Image augmentation and preprocessing
- Transfer learning with MobileNet
- Model training and checkpointing
- Training history visualization
- Sample image classification
- Confusion matrix and classification report generation

The model classifies complete images; it does not generate bounding boxes or locate affected leaf areas.

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- OpenCV
- scikit-learn
- Jupyter Notebook / Google Colab

## Supported Classes

| Index | Class |
|------:|-------|
| 0 | Bacterial Spot |
| 1 | Early Blight |
| 2 | Late Blight |
| 3 | Leaf Mold |
| 4 | Septoria Leaf Spot |
| 5 | Two-Spotted Spider Mite |
| 6 | Target Spot |
| 7 | Tomato Yellow Leaf Curl Virus |
| 8 | Tomato Mosaic Virus |
| 9 | Healthy |

The notebook obtains the class mapping from `train_data.class_indices`.

## Model Architecture

The model uses a pretrained MobileNet backbone with a custom classification head.

| Component | Configuration |
|-----------|---------------|
| Backbone | MobileNet with ImageNet weights |
| Input size | 224 × 224 × 3 |
| Original classification head | Removed |
| Backbone layers | Frozen |
| Custom head | Flatten → Dense |
| Output layer | 10 units with softmax |
| Optimizer | Adam |
| Loss function | Categorical cross-entropy |
| Batch size | 64 |
| Training epochs | 5 |

### Image Preprocessing

The notebook uses:

- Pixel rescaling by `1/255`
- Random zoom
- Shear transformations
- Horizontal flipping

Training callbacks include:

- **EarlyStopping:** Monitors validation accuracy.
- **ModelCheckpoint:** Saves the best checkpoint as `best_model.h5`.

## Dataset

The dataset is downloaded separately and is not included in the repository.

The notebook contains a [Dropbox dataset link](https://www.dropbox.com/s/en2g7ml1m5a8yrd/Datasets.zip?dl=1). Its current availability has not been verified.

The saved notebook output reports **18,345 training images across 10 classes**.

Expected directories:

- `Datasets/train/` — training images organized by class
- `Datasets/valid/` — validation images organized by class
- `Datasets/test/` — sample images for prediction

Dataset provenance and licensing should be verified before reuse.

## Repository Contents

| File | Description |
|------|-------------|
| `Tomato leaves Disease Detection.ipynb` | Dataset preparation, model training, visualization, and inference |

The dataset, trained model files, and a web interface are not included.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/SumairSalam/Tomato_leaves-Detection-AI_project.git
cd Tomato_leaves-Detection-AI_project
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install tensorflow numpy pandas matplotlib opencv-python scikit-learn notebook
```

The notebook uses older Keras APIs and does not pin dependency versions. Compatibility updates may be required.

### 4. Open the Notebook

```bash
jupyter notebook "Tomato leaves Disease Detection.ipynb"
```

Alternatively, upload the notebook to Google Colab. A GPU can help reduce training time.

## Running the Project

Before running all cells:

1. Review the cleanup cells, which delete directories and ZIP files.
2. Download and extract the dataset.
3. Update dataset paths for your environment.
4. Correct the validation directory as described below.
5. Run the preprocessing, model creation, and training cells.
6. Review the training plots.
7. Update the sample image path and run prediction.

The pretrained MobileNet weights require an initial download unless already cached.

## Required Notebook Corrections

### Validation Directory

The current validation generator reads the training directory:

```python
directory="/content/Datasets/train"
```

Change it to the separate validation directory:

```python
val_data = val_datagen.flow_from_directory(
    directory="/content/Datasets/valid",
    target_size=(224, 224),
    batch_size=64,
    class_mode="categorical",
    shuffle=False
)
```

Using `shuffle=False` keeps predictions aligned with class labels when generating the confusion matrix and classification report.

### Training API

Replace the deprecated call:

```python
model.fit_generator(...)
```

with:

```python
hist = model.fit(
    train_data,
    epochs=5,
    validation_data=val_data,
    callbacks=[es, mc]
)
```

If required by your environment, replace:

```python
train_data.next()
```

with:

```python
next(train_data)
```

For newer environments, use consistent `tensorflow.keras` imports and import image utilities from `tensorflow.keras.utils`.

## Recorded Results

The saved five-epoch training run reports:

| Metric | Recorded Value |
|--------|---------------:|
| Final training accuracy | 95.15% |
| Highest reported validation accuracy | 97.05% |

**Evaluation caveat:** Both generators currently load images from the training directory. Therefore, the recorded 97.05% does not represent performance on independent validation data.

The saved classification report also shows approximately 10% accuracy. Its predictions are compared against fixed-order labels while the generator shuffles images, making that report unreliable.

Correct the validation setup and rerun evaluation on held-out images before reporting final performance. These historical results have not been independently reproduced.

## Predicting a Leaf Condition

After training or loading a compatible model, the notebook predicts a class using:

```python
path = "/content/Datasets/test/EarlyBlight.png"

img = load_img(path, target_size=(224, 224))
image_array = img_to_array(img) / 255.0
input_array = np.expand_dims(image_array, axis=0)

class_mapping = {
    index: name
    for name, index in train_data.class_indices.items()
}

probabilities = model.predict(input_array)
predicted_index = np.argmax(probabilities[0])

print("Predicted class:", class_mapping[predicted_index])

plt.imshow(image_array)
plt.title(class_mapping[predicted_index])
plt.axis("off")
plt.show()
```

Change `path` to the image you want to classify.

The saved notebook example predicts `Tomato___Early_blight`.

## Saved Models

The notebook creates:

- `best_model.h5` — best checkpoint according to validation accuracy
- `model_inception.h5` — model saved by a separate export cell

Despite the second filename, the implemented architecture is **MobileNet**, not Inception.

## Future Improvements

- Evaluate on separate training, validation, and test sets.
- Save class mappings with the trained model.
- Standardize preprocessing for the pretrained backbone and retrain consistently.
- Add reproducible random seeds and pinned dependencies.
- Generate correctly aligned per-class evaluation metrics.
- Test on field images with varied lighting and backgrounds.
- Add an image upload interface.
- Explore fine-tuning selected MobileNet layers.

## Author

[Sumair Salam](https://github.com/SumairSalam)
