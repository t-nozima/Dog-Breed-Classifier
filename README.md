# Dog Breed Classifier
Deep Learning Model for Classifying 20 Dog Breeds

## Overview 

Dog Breed Classifier is a deep learning project designed to classify images of dogs into 20 different breeds.
The model is built using a custom CNN architecture with Residual Blocks (ResNet-style).

The project also includes a fully interactive Streamlit web demo (demo.py) where users can upload an image and instantly receive a breed prediction.

## Project Summary

This work implements a deep convolutional neural network with residual connections to prevent vanishing gradients and improve feature extraction.
Key techniques used during training:

- Dataset standardization
- Heavy data augmentation
- Class weighting to handle imbalance
- Custom preprocessing (BGR, ImageNet mean/std normalization)

## Streamlit Demo (demo.py)

The project includes a clean, modern Streamlit application with gradient UI, custom CSS, centered layout, and formatted breed predictions.

## How to Launch the Demo
### Install dependencies:
```pip install -r requirements.txt```

### Ensure the trained model file exists:
`model.h5` must be located in the same directory as `demo.py`:

```
project/
├── demo.py
├── model.h5
└── ...
```

### Start the Streamlit web app

To launch the application locally, run the following command:

```bash
streamlit run demo.py
```

## What `demo.py` Does

The web application:

- Lets the user upload an image (**jpg, jpeg, png**)

### Preprocessing
- Converts to **RGB**
- Resizes to **224 × 224**
- Converts **RGB → BGR**
- Applies **ImageNet mean/std normalization**

### Loads the model via
```python
tf.keras.models.load_model("model.h5")
```

### Prediction
- Predicts the breed  
- Formats class names (**underscores → spaces**)

### Displays
- Predicted breed  
- Confidence score  
- Centered preview image  
- Styled result card  

### UI Details
- Gradient title text  
- Transparent uploader  
- Animated buttons  
- Shadowed result card  
- Custom **“Outfit”** font  
- Fully centered layout  


### Dataset directory must be:
```swift
/Team_39/Images/
```

### Dataset

Dataset must be located here:

```swift
/Team_39/Images/
```

Download source:

https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset

## Technologies

A set of tools and libraries used in this project:

- **TensorFlow / Keras** — deep learning model training  
- **NumPy** — numerical operations  
- **Pandas** — data handling and preprocessing  
- **Matplotlib** — plotting training curves  
- **Seaborn** — confusion matrix visualization  
- **scikit-learn** — metrics & evaluation tools  
- **Streamlit** — interactive model demo  
- **OpenCV** — image processing  
- **Pillow (PIL)** — image loading & format conversion

## Recommended Project Structure

```
├── demo.py                      # Streamlit demo app
├── model.h5                     # Trained CNN model
├── dog_breed_classifier.ipynb   # Training & evaluation notebook
├── requirements.txt             # Project dependencies
├── Team_39/
│    └── Images/                 # Dataset directory
└── README.md                    # Project documentation
```

## Full Documentation

Detailed project documentation is available on Notion:

**https://www.notion.so/Dog-Breed-Classifier-2c611bcd8ed78051bccfd5f3b9142321**

---

## Authors

**Team 39 — NxD Project**

---

## License

This project is licensed under the **MIT License**.















