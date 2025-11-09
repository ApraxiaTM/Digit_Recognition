# Handwritten Digit Recognition using CNN

A deep learning application for recognizing handwritten digits (0-9) using Convolutional Neural Networks (CNN) trained on the MNIST dataset.

![Handwritten Digit Recognition Demo](images/demo.png)

## Table of Contents
- [Features](#features)
- [Theory](#theory)
  - [Introduction](#introduction)
  - [Dataset: MNIST](#dataset-mnist)
  - [Convolutional Neural Networks (CNN)](#convolutional-neural-networks-cnn)
  - [Architecture](#our-cnn-architecture)
  - [Training Process](#training-process)
  - [Image Preprocessing](#image-preprocessing)
  - [Applications](#real-world-applications)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

## Features
- ✅ CNN model with 99%+ accuracy
- ✅ Interactive drawing interface
- ✅ Real-time digit prediction with confidence scores
- ✅ Comprehensive visualization and analysis
- ✅ Upload custom images for prediction
- ✅ Complete training and evaluation pipeline

---

## Theory

### Introduction

Handwritten digit recognition is a classic computer vision problem where the goal is to classify images of handwritten digits (0-9) into their respective numerical classes. This application demonstrates the power of deep learning in pattern recognition tasks.

The project uses deep learning techniques, specifically Convolutional Neural Networks (CNNs), to achieve high accuracy in recognizing handwritten digits. This technology forms the foundation for many real-world applications including automated form processing, check reading, and postal mail sorting.

### Dataset: MNIST

**MNIST** (Modified National Institute of Standards and Technology) is the benchmark dataset used in this project:

- **Training Set**: 60,000 images
- **Test Set**: 10,000 images
- **Image Size**: 28×28 pixels
- **Color**: Grayscale (0-255 pixel values)
- **Classes**: 10 digits (0-9)

MNIST is widely used in machine learning research and education due to its:
- Manageable size for quick training
- Well-balanced class distribution
- Standardized format
- Proven benchmark for comparing algorithms

### Convolutional Neural Networks (CNN)

#### Why CNN for Image Recognition?

Traditional neural networks treat images as flat vectors, which has several limitations:
- Loss of spatial information
- No understanding of pixel relationships
- Requires massive number of parameters
- Poor generalization to variations

**CNNs solve these problems by:**
- Preserving spatial relationships between pixels
- Using convolution operations to detect local features
- Sharing weights across the image (parameter efficiency)
- Building hierarchical feature representations

#### CNN Architecture Components

##### 1. Convolutional Layers
Convolutional layers are the core building blocks of CNNs:

- **Purpose**: Extract features from input images
- **Operation**: Apply learnable filters/kernels across the image
- **Feature Detection**: Each filter detects specific patterns (edges, curves, textures)
- **Output**: Feature maps highlighting detected patterns

**Mathematical Formula:**
```
Output Size = (Input Size - Filter Size + 2×Padding) / Stride + 1
```

**Key Concepts:**
- **Filters/Kernels**: Small matrices (e.g., 3×3) that slide over the image
- **Stride**: Step size for moving the filter
- **Padding**: Adding borders to preserve spatial dimensions

##### 2. Activation Functions (ReLU)

**ReLU (Rectified Linear Unit)**: `f(x) = max(0, x)`

**Benefits:**
- Introduces non-linearity to the network
- Enables learning of complex patterns
- Computationally efficient
- Mitigates vanishing gradient problem
- Sparse activation (only positive values pass through)

##### 3. Pooling Layers (MaxPooling)

Pooling layers reduce spatial dimensions while retaining important information:

**MaxPooling Operation:**
- Divides input into rectangular regions
- Outputs the maximum value from each region
- Typical size: 2×2 with stride 2 (reduces dimensions by half)

**Benefits:**
- Reduces computational cost
- Provides translation invariance
- Controls overfitting
- Retains dominant features

##### 4. Dropout

Dropout is a regularization technique:

**How it works:**
- Randomly deactivates a percentage of neurons during training
- Forces the network to learn redundant representations
- Prevents co-adaptation of neurons

**Benefits:**
- Reduces overfitting
- Improves generalization
- Acts as ensemble learning

##### 5. Fully Connected (Dense) Layers

Dense layers connect every neuron to every neuron in the next layer:

**Purpose:**
- Combine extracted features for classification
- Learn complex decision boundaries
- Final layer uses softmax for probability distribution

**Softmax Function:**
```
softmax(x_i) = exp(x_i) / Σ(exp(x_j))
```
Converts raw scores to probabilities that sum to 1.

### Our CNN Architecture

```
Input Layer (28×28×1)
    ↓
Convolutional Layer 1
├─ 32 filters (3×3)
├─ ReLU activation
└─ Output: 26×26×32
    ↓
MaxPooling Layer 1
├─ Pool size: 2×2
└─ Output: 13×13×32
    ↓
Convolutional Layer 2
├─ 64 filters (3×3)
├─ ReLU activation
└─ Output: 11×11×64
    ↓
MaxPooling Layer 2
├─ Pool size: 2×2
└─ Output: 5×5×64
    ↓
Convolutional Layer 3
├─ 64 filters (3×3)
├─ ReLU activation
└─ Output: 3×3×64
    ↓
Flatten Layer
└─ Output: 576 neurons
    ↓
Dense Layer 1
├─ 64 neurons
├─ ReLU activation
└─ Dropout (0.5)
    ↓
Output Layer (Dense)
├─ 10 neurons (one per digit)
├─ Softmax activation
└─ Output: Probability distribution over 10 classes
```

**Architecture Rationale:**
- **Progressive feature extraction**: 32 → 64 → 64 filters
- **Spatial reduction**: 28×28 → 13×13 → 5×5 → 3×3
- **Feature complexity**: Low-level → Mid-level → High-level features
- **Regularization**: Dropout prevents overfitting

### Training Process

#### Loss Function: Categorical Cross-Entropy

Measures the difference between predicted and true probability distributions:

**Formula:**
```
L = -Σ(y_true × log(y_pred))
```

Where:
- `y_true`: One-hot encoded true label
- `y_pred`: Predicted probability distribution
- Lower loss = better predictions

**Why Cross-Entropy?**
- Suitable for multi-class classification
- Penalizes confident wrong predictions heavily
- Smooth gradient for optimization

#### Optimizer: Adam

**Adam** (Adaptive Moment Estimation) combines the best properties of AdaGrad and RMSProp:

**Key Features:**
- Adaptive learning rates for each parameter
- Momentum-based updates
- Bias correction for initial estimates
- Efficient and requires minimal tuning

**Hyperparameters:**
- Learning rate: 0.001 (default)
- β₁ = 0.9 (exponential decay rate for first moment)
- β₂ = 0.999 (exponential decay rate for second moment)

#### Training Configuration

```python
Epochs: 10
Batch Size: 128
Validation Split: 10%
Optimizer: Adam
Loss: Categorical Cross-Entropy
Metrics: Accuracy
```

#### Evaluation Metrics

**Accuracy**: Percentage of correctly classified images
```
Accuracy = (Correct Predictions) / (Total Predictions) × 100%
```

**Confusion Matrix**: Shows classification performance per class

### Image Preprocessing

Proper preprocessing is crucial for model performance:

#### 1. Normalization
```python
normalized_image = pixel_values / 255.0
```
- Converts pixel range from [0, 255] to [0, 1]
- Helps gradient descent converge faster
- Prevents numerical instability

#### 2. Reshaping
```python
reshaped_image = image.reshape(28, 28, 1)
```
- Adds channel dimension for CNN input
- Format: (height, width, channels)

#### 3. One-Hot Encoding
```python
# Label 5 becomes: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
```
- Converts integer labels to binary vectors
- Required for categorical cross-entropy loss

#### 4. Color Inversion (if needed)
```python
inverted_image = 255 - image
```
- MNIST uses white digits on black background
- Some drawing tools use opposite convention

### Real-World Applications

This technology powers numerous practical applications:

#### Financial Services
- **Automated Check Processing**: Reading handwritten amounts and signatures
- **Form Digitization**: Converting paper forms to digital data
- **Fraud Detection**: Analyzing signature patterns

#### Postal Services
- **Mail Sorting**: Reading ZIP codes and addresses
- **Package Routing**: Automated address recognition
- **Delivery Optimization**: Digitizing handwritten labels

#### Education
- **Automated Grading**: Recognizing handwritten answers
- **Learning Tools**: Interactive math problem solvers
- **Accessibility**: Converting handwritten notes to digital text

#### Healthcare
- **Prescription Reading**: Digitizing doctor's prescriptions
- **Medical Form Processing**: Converting patient forms
- **Record Digitization**: Archiving handwritten medical records

#### Government
- **Census Data Entry**: Processing handwritten forms
- **Tax Form Processing**: Reading handwritten tax returns
- **Document Archival**: Digitizing historical documents

---

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager
- Jupyter Notebook

### Step 1: Clone the Repository
```bash
git clone https://github.com/ApraxiaTM/Digit_Recognition.git
cd handwritten-digit-recognition
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Launch Jupyter Notebook
```bash
jupyter notebook digit_recognition.ipynb
```

## Usage

### Training the Model

1. Open `digit_recognition.ipynb` in Jupyter Notebook
2. Run all cells sequentially (Cell → Run All)
3. The model will automatically:
   - Load the MNIST dataset
   - Preprocess the data
   - Build the CNN architecture
   - Train for 10 epochs
   - Evaluate on test set
   - Display results and visualizations

### Making Predictions

#### Method 1: Interactive Drawing Interface
```python
# Run the drawing interface cell
drawing_app = DrawingCanvas(model)
drawing_app.display()
```
- Draw a digit on the canvas
- Click "Predict" to see results
- Click "Clear" to draw again

#### Method 2: Upload Custom Image
```python
# Predict from uploaded image
predict_uploaded_image('path/to/your/image.png')
```

#### Method 3: Test on MNIST Samples
```python
# Automatically tests on random samples from test set
# Results displayed with confidence scores
```

### Loading Pre-trained Model

```python
from tensorflow import keras

# Load saved model
model = keras.models.load_model('mnist_cnn_model.h5')

# Make predictions
predictions = model.predict(your_image)
```

## Results

### Model Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 99.2% |
| **Training Time** | ~10 minutes (CPU) |
| **Model Size** | ~3 MB |
| **Parameters** | ~93,000 |
| **Inference Time** | <10ms per image |

### Training History

The model achieves:
- **Training Accuracy**: 99.5%
- **Validation Accuracy**: 99.2%
- **Test Accuracy**: 99.2%
- **Minimal Overfitting**: Validation and training curves closely aligned

### Sample Predictions

The notebook includes visualizations showing:
- Correctly classified digits with confidence scores
- Misclassified examples (rare cases)
- Probability distributions for each prediction
- Confusion matrix analysis

## Project Structure

```
handwritten-digit-recognition/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── mnist_digit_recognition.ipynb      # Main Jupyter notebook
│
├── images/                            # Screenshots and visualizations
│   ├── demo.png
│   ├── architecture.png
│   ├── results.png
│   └── samples.png
│
├── models/                            # Saved models
│   ├── mnist_cnn_model.h5            # Keras model file
│   └── mnist_cnn_model/              # SavedModel format
│
└── test_images/                       # Sample test images
    ├── digit_0.png
    ├── digit_1.png
    └── ...
```

## Technologies Used

- **TensorFlow/Keras**: Deep learning framework
- **NumPy**: Numerical computations
- **Matplotlib**: Data visualization
- **OpenCV**: Image processing
- **Pillow**: Image manipulation
- **IPyWidgets**: Interactive widgets for Jupyter

## Model Architecture Details

```python
Total params: 93,322
Trainable params: 93,322
Non-trainable params: 0
```

### Layer-by-Layer Breakdown

| Layer | Output Shape | Parameters |
|-------|--------------|------------|
| Conv2D (32 filters) | (26, 26, 32) | 320 |
| MaxPooling2D | (13, 13, 32) | 0 |
| Conv2D (64 filters) | (11, 11, 64) | 18,496 |
| MaxPooling2D | (5, 5, 64) | 0 |
| Conv2D (64 filters) | (3, 3, 64) | 36,928 |
| Flatten | (576,) | 0 |
| Dense (64 units) | (64,) | 36,928 |
| Dropout | (64,) | 0 |
| Dense (10 units) | (10,) | 650 |

## Future Enhancements

- [ ] Add support for multi-digit recognition
- [ ] Implement data augmentation for improved robustness
- [ ] Create web application with Flask/Django
- [ ] Add support for other languages (Arabic, Chinese digits)
- [ ] Implement transfer learning with pre-trained models
- [ ] Add real-time webcam digit recognition
- [ ] Create mobile app version
- [ ] Implement ensemble methods for higher accuracy

## Troubleshooting

### Common Issues

**Issue**: Model accuracy is low
- **Solution**: Ensure data is properly normalized and preprocessed

**Issue**: Drawing interface not working
- **Solution**: Install ipywidgets: `pip install ipywidgets`
- Enable widgets: `jupyter nbextension enable --py widgetsnbextension`

**Issue**: TensorFlow installation errors
- **Solution**: Use specific version: `pip install tensorflow==2.10.0`

**Issue**: Out of memory during training
- **Solution**: Reduce batch size in training configuration

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- MNIST dataset creators: Yann LeCun, Corinna Cortes, and Christopher J.C. Burges
- TensorFlow and Keras teams for excellent deep learning frameworks
- The open-source community for inspiration and resources

## References

1. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition.
2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
3. TensorFlow Documentation: https://www.tensorflow.org/
4. MNIST Database: http://yann.lecun.com/exdb/mnist/

## Author

```bibtex
@misc{handwritten_digit_recognition,
  author = {Winston Narada Kusumahadi},
  title = {Handwritten Digit Recognition using CNN},
  year = {2025},
  publisher = {GitHub},
}
```

---

**⭐ If you find this project helpful, please consider giving it a star!**

**📧 Questions or suggestions? Feel free to open an issue or contact me directly.**

---

*Last Updated: November 2025*