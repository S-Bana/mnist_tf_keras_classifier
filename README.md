
# MNIST Image Classifier with TensorFlow & Keras

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

A simple image classifier for handwritten digits using the MNIST dataset, built with TensorFlow and Keras. This project covers data preprocessing, model building, training, evaluation, and prediction.

## Features

- Loads and preprocesses the MNIST dataset
- Builds a neural network model using Keras
- Trains and evaluates the model
- Makes predictions on new data

## Installation

1.  **Clone the repository:**
    ```bash
    git clone git@github.com:S-Bana/mnist_tf_keras_classifier.git
    cd mnist_tf_keras_classifier
    ```
2.  **Install required libraries:**
    ```bash
    pip3 install -r requirements.txt
    ```
    *Note: A `requirements.txt` file would typically list `numpy`, `tensorflow`, and `pandas` (though pandas isn't directly used).*
3.  **Ensure Dataset Availability:**
    * The script expects the MNIST dataset to be available locally at `dataset/mnist.npz`.
    * If you don't have it, uncomment the `datasets.mnist.load_data()` and `np.savez(...)` lines in `mnist_classifier.py` once to download and save it. Remember to re-comment them after the first run if you wish to use the local file.
4.  **Run the classifier:**
    ```bash
    python3 mnist_classifier.py
    ```

## Project Structure

| File/Folder         | Description                                |
| :------------------ | :----------------------------------------- |
| `mnist_classifier.py` | Main script for training & testing the CNN |
| `README.md`         | Project documentation and concept explanations |
| `LICENSE`           | GNU GPL v3 License details                 |
| `dataset/`          | Directory to store the `mnist.npz` dataset |
| `requirements.txt`  | Lists Python dependencies                  |

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.

---

## MNIST Digit Classifier using Convolutional Neural Networks (CNN) - In-depth Explanation

This section provides a detailed explanation of the Python code and the core machine learning concepts applied in this project.

### Table of Contents (for detailed explanation)

- [Introduction to the CNN Project](#introduction-to-the-cnn-project)
- [Dependencies](#dependencies)
- [Code Explanation](#code-explanation)
  - [1. Import Libraries](#1-import-libraries)
  - [2. Load Dataset](#2-load-dataset)
  - [3. Preprocessing](#3-preprocessing)
  - [4. Build CNN Model](#4-build-cnn-model)
  - [5. Compile Model](#5-compile-model)
  - [6. Train Model](#6-train-model)
  - [7. Evaluate Model](#7-evaluate-model)
  - [8. Make Predictions](#8-make-predictions)
- [Key Concepts Explained](#key-concepts-explained)
  - [Convolutional Neural Networks (CNNs)](#convolutional-neural-networks-cnns)
  - [Layers](#layers)
    - [Conv2D Layer](#conv2d-layer)
    - [MaxPooling2D Layer](#maxpooling2d-layer)
    - [Flatten Layer](#flatten-layer)
    - [Dense Layer](#dense-layer)
  - [Activation Functions](#activation-functions)
    - [ReLU (Rectified Linear Unit)](#relu-rectified-linear-unit)
    - [Softmax](#softmax)
    - [Other Common Activation Functions](#other-common-activation-functions)
  - [Loss Functions](#loss-functions)
    - [Categorical Crossentropy](#categorical-crossentropy)
    - [Other Common Loss Functions](#other-common-loss-functions)
  - [Optimizers](#optimizers)
    - [Adam Optimizer](#adam-optimizer)
    - [Other Common Optimizers](#other-common-optimizers)
  - [Metrics](#metrics)
    - [Accuracy](#accuracy)
  - [Epochs](#epochs)
  - [Batch Size](#batch-size)
  - [One-Hot Encoding](#one-hot-encoding)
  - [Normalization](#normalization)

---

### Introduction to the CNN Project

This project aims to build and train a Convolutional Neural Network (CNN) to recognize handwritten digits (0-9) from the MNIST dataset. The MNIST dataset is a classic benchmark in machine learning and computer vision, often used as the "hello world" for deep learning.

### Dependencies

You need the following Python libraries installed:

-   `numpy`
-   `tensorflow` (which includes Keras)
-   `pandas` (imported in the script, though not actively used for this specific task)

You can install them using pip as shown in the [Installation](#installation) section.

## Code Explanation

Below is a detailed breakdown of the `mnist_classifier.py` script.

### 1. Import Libraries

```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
````

  - **`numpy as np`**: Fundamental package for numerical computation in Python, especially useful for array operations.
  - **`pandas as pd`**: Data manipulation and analysis library. Although imported here, it's not directly used in the current version of the script.
  - **`tensorflow as tf`**: The core open-source machine learning framework developed by Google. It provides a comprehensive ecosystem for building and deploying ML models.
  - **`tensorflow.keras`**: Keras is a high-level API for building and training deep learning models, now integrated directly into TensorFlow.
      - **`datasets`**: Contains functions to load popular datasets (like MNIST).
      - **`layers`**: Provides common neural network layers (e.g., `Conv2D`, `MaxPooling2D`, `Dense`).
      - **`models`**: Used to define model architectures (e.g., `Sequential` model).
  - **`tensorflow.keras.utils.to_categorical`**: A utility function to convert integer class labels into one-hot encoded vectors.

### 2\. Load Dataset

```python
# (train_img, train_label), (test_img, test_label) = datasets.mnist.load_data()
# np.savez('dataset/mnist.npz',
#          train_img=train_img,
#          train_label=train_label,
#          test_img=test_img,
#          test_label=test_label)

path = 'dataset/mnist.npz'

with np.load(path, allow_pickle=True) as data:
    train_img = data['x_train']
    train_label = data['y_train']
    test_img = data['x_test']
    test_label = data['y_test']
```

  - The commented-out lines show how to download the MNIST dataset directly from Keras and how to save it to a local `.npz` file.
  - The active code loads the dataset from a pre-saved `.npz` file located at `dataset/mnist.npz`. This is useful for offline training or if you want to ensure consistent data loading.
  - `train_img`, `train_label`: Training images and their corresponding labels.
  - `test_img`, `test_label`: Test images and their corresponding labels (used for evaluation).

### 3\. Preprocessing

```python
# Normalize the pixel value to be between 0 and 1
train_img = train_img / 255.0
test_img = test_img / 255.0

print(train_img.shape) # Expected: (60000, 28, 28)
print(test_img.shape)  # Expected: (10000, 28, 28)

# Reshape images to (height, width, channels) format for Conv2D layers
train_img = train_img.reshape((train_img.shape[0], 28, 28, 1))
test_img = test_img.reshape((test_img.shape[0], 28, 28, 1))

print(train_img.shape) # Expected: (60000, 28, 28, 1)
print(test_img.shape)  # Expected: (10000, 28, 28, 1)

# Convert labels to one-hot encoding
train_label = to_categorical(train_label)
test_label = to_categorical(test_label)
```

  - **Normalization**: Pixel values in grayscale images typically range from 0 to 255. Dividing by `255.0` scales these values to a floating-point range of 0 to 1. This helps in faster convergence during training and prevents larger input values from disproportionately affecting the weights.
  - **Reshaping**: Convolutional layers in Keras (`Conv2D`) expect input images to be in the format `(batch_size, height, width, channels)`.
      - MNIST images are 28x28 pixels.
      - They are grayscale, meaning they have 1 channel.
      - The `reshape` operation adds this channel dimension. So, `(60000, 28, 28)` becomes `(60000, 28, 28, 1)` for `train_img`, and similarly for `test_img`.
  - **One-Hot Encoding**: Converts integer labels (e.g., 0, 1, 2, ..., 9) into a binary vector representation. For example, if the label is `3`, it becomes `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`. This is required for the `categorical_crossentropy` loss function used in multi-class classification.

### 4\. Build CNN Model

```python
# Initialize a sequential model
model = models.Sequential()

# Add layers to the model
# Input layer
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))

# Hidden layers
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))

# Output layer (10 classes for digits 0-9)
model.add(layers.Dense(10, activation='softmax'))
```

  - **`models.Sequential()`**: This is the simplest type of Keras model, where layers are stacked linearly.
  - **`model.add(...)`**: Adds a layer to the model.
      - **`layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))`**:
          - First convolutional layer.
          - `32`: Number of filters (feature maps) the layer will learn.
          - `(3, 3)`: Size of the convolutional kernel (filter).
          - `activation='relu'`: Applies the ReLU activation function (see [Activation Functions](https://www.google.com/search?q=%23activation-functions) below).
          - `input_shape=(28, 28, 1)`: Specifies the shape of the input data for the *first* layer. This corresponds to 28x28 pixel grayscale images.
      - **`layers.MaxPooling2D((2, 2))`**:
          - Downsamples the feature maps by taking the maximum value in 2x2 non-overlapping regions.
          - This reduces the spatial dimensions of the output, reducing computational cost and making the model more robust to small translations in the input.
      - Subsequent `Conv2D` and `MaxPooling2D` layers: These form the "feature learning" part of the CNN, extracting increasingly complex features from the images. The number of filters often increases in deeper layers (e.g., from 32 to 64).
      - **`layers.Flatten()`**: Converts the 3D output from the last convolutional/pooling layer into a 1D vector. This is necessary because fully connected (`Dense`) layers expect 1D input.
      - **`layers.Dense(64, activation='relu')`**:
          - A fully connected (dense) hidden layer with 64 neurons.
          - `activation='relu'`: Applies the ReLU activation function.
      - **`layers.Dense(10, activation='softmax')`**:
          - The output layer.
          - `10`: Number of neurons, corresponding to the 10 possible digit classes (0-9).
          - `activation='softmax'`: Applies the Softmax activation function (see [Activation Functions](https://www.google.com/search?q=%23activation-functions) below). This converts the raw output scores into a probability distribution over the 10 classes, where the sum of probabilities is 1.

### 5\. Compile Model

```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

  - **`model.compile()`**: Configures the model for training.
      - **`optimizer='adam'`**: Specifies the optimization algorithm used to update the model's weights during training. Adam is a popular and generally efficient choice (see [Optimizers](https://www.google.com/search?q=%23optimizers) below).
      - **`loss='categorical_crossentropy'`**: Defines the loss function that the model will try to minimize. This is suitable for multi-class classification when labels are one-hot encoded (see [Loss Functions](https://www.google.com/search?q=%23loss-functions) below).
      - **`metrics=['accuracy']`**: Specifies the metrics to be evaluated by the model during training and testing. `accuracy` measures the proportion of correctly classified samples (see [Metrics](https://www.google.com/search?q=%23metrics) below).

### 6\. Train Model

```python
model.fit(train_img, train_label, epochs=5, batch_size=64, validation_data=(test_img, test_label))
```

  - **`model.fit()`**: Trains the model on the provided data.
      - **`train_img`, `train_label`**: The input features and corresponding target labels for training.
      - **`epochs=5`**: The number of times the training algorithm will iterate over the entire training dataset. One epoch means that each sample in the training dataset has had an opportunity to update the model's internal parameters.
      - **`batch_size=64`**: The number of samples processed before the model's parameters are updated. Using mini-batches makes training more stable and efficient than processing one sample at a time (stochastic gradient descent) or the entire dataset at once (batch gradient descent).
      - **`validation_data=(test_img, test_label)`**: Data on which to evaluate the loss and any model metrics at the end of each epoch. This helps monitor for overfitting – if validation loss starts increasing while training loss decreases, the model might be overfitting.

### 7\. Evaluate Model

```python
loss_, acc_ = model.evaluate(test_img, test_label)
print(f'Test Accuracy: {acc_*100:.3}% \t Test Loss: {loss_}')
```

  - **`model.evaluate()`**: Assesses the model's performance on the provided test data.
      - It returns the loss value and the values of the metrics specified during compilation (in this case, accuracy).
  - The output shows the final accuracy and loss on the unseen test set.

### 8\. Make Predictions

```python
predict_ = model.predict(test_img)
print(f'Prediction for first image: {np.argmax(predict_[0])}')
```

  - **`model.predict()`**: Generates output predictions for the input samples.
      - `predict_` will be an array where each row is a probability distribution over the 10 classes for a corresponding input image. For example, `predict_[0]` might look like `[0.01, 0.005, ..., 0.95, ..., 0.002]`, indicating a high probability for one specific digit.
  - **`np.argmax(predict_[0])`**: This function returns the index of the maximum value in the array `predict_[0]`. Since the output layer uses `softmax` and represents probabilities for each digit, the index with the highest probability is the model's predicted digit for the first test image.

-----

## Key Concepts Explained

### Convolutional Neural Networks (CNNs)

**Convolutional Neural Networks (CNNs)** are a class of deep neural networks, most commonly applied to analyzing visual imagery. They are particularly effective for tasks like image classification, object detection, and segmentation.

**Why CNNs for Images?**
Traditional neural networks struggle with images because:

1.  **High Dimensionality:** Even small images (e.g., 28x28 pixels) result in many input features (784). Larger images lead to an explosion in parameters, making models computationally expensive and prone to overfitting.
2.  **Spatial Information Loss:** Flattening an image (converting 2D to 1D) discards valuable spatial relationships between pixels.
3.  **Lack of Translation Invariance:** A traditional network might need to learn the same feature (e.g., an edge) at different locations in the image independently.

CNNs address these issues through:

  - **Sparse Connectivity/Local Receptive Fields:** Neurons in a convolutional layer are only connected to a small, localized region of the input (the "receptive field"), reducing the number of parameters.
  - **Parameter Sharing:** The same set of weights (a "filter" or "kernel") is applied across the entire input. This drastically reduces parameters and allows the network to detect the same feature regardless of its position in the image (translation invariance).
  - **Pooling Layers:** Periodically downsample the feature maps, reducing dimensionality and making the model more robust to minor distortions.

The architecture typically consists of alternating **convolutional layers** (for feature extraction) and **pooling layers** (for dimensionality reduction), followed by **fully connected (Dense) layers** for classification.

### Layers

Layers are the fundamental building blocks of neural networks. They process input data and transform it into output data.

#### Conv2D Layer

  - **Full Definition:** `tf.keras.layers.Conv2D(filters, kernel_size, activation=None, input_shape=None, ...)`
  - **Purpose:** The core building block of CNNs for image processing. It performs a *convolution* operation on the input.
  - **How it Works:** A `Conv2D` layer applies a set of learnable filters (also called kernels) to the input image. Each filter slides across the width and height of the input volume, computing the dot product between the filter's weights and the input pixels at each position. This produces a 2D activation map (or feature map) that indicates where the filter's specific feature is present in the input.
      - **`filters`**: The number of output feature maps (i.e., the number of different patterns/features the layer will learn to detect). More filters allow the network to learn more diverse features.
      - **`kernel_size`**: A tuple specifying the height and width of the 2D convolution window (e.g., `(3, 3)` for a 3x3 filter).
      - **`activation`**: The activation function to apply after the convolution (e.g., `'relu'`).
      - **`input_shape`**: Required only for the first layer in a `Sequential` model, defining the shape of a single input sample (e.g., `(28, 28, 1)` for a 28x28 grayscale image).
  - **Why Used:** To automatically learn hierarchical features from images. Early layers might learn simple features like edges and corners, while deeper layers combine these to learn more complex patterns (e.g., eyes, noses in faces, or parts of digits).

#### MaxPooling2D Layer

  - **Full Definition:** `tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', ...)`
  - **Purpose:** A type of pooling layer that reduces the spatial dimensions (width and height) of the input feature maps.
  - **How it Works:** It divides the input into non-overlapping rectangular pooling regions (defined by `pool_size`) and, for each region, outputs the maximum value.
      - **`pool_size`**: A tuple specifying the size of the pooling window (e.g., `(2, 2)` means a 2x2 window).
      - **`strides`**: The step size of the pooling window. If `None`, it defaults to `pool_size`, meaning non-overlapping windows.
  - **Why Used:**
    1.  **Dimensionality Reduction:** Reduces the number of parameters and computational cost, preventing overfitting.
    2.  **Translation Invariance:** Makes the model more robust to small shifts or distortions in the input image. If a feature (like an edge) shifts slightly within a pooling window, the maximum value will still be detected, leading to the same output.
    3.  **Feature Abstraction:** Highlights the most prominent features within each region.

#### Flatten Layer

  - **Full Definition:** `tf.keras.layers.Flatten()`
  - **Purpose:** Reshapes the input into a 1D array (vector).
  - **How it Works:** It takes the multi-dimensional output from previous layers (e.g., `(batch_size, height, width, channels)` from `Conv2D` or `MaxPooling2D`) and transforms it into a flat, one-dimensional feature vector `(batch_size, height * width * channels)`.
  - **Why Used:** To transition from the convolutional/pooling layers (which operate on spatial data) to traditional fully connected (`Dense`) layers, which expect a 1D input vector.

#### Dense Layer

  - **Full Definition:** `tf.keras.layers.Dense(units, activation=None, ...)`
  - **Purpose:** A standard fully connected neural network layer.
  - **How it Works:** Each neuron in a `Dense` layer is connected to every neuron in the previous layer. It performs a linear transformation on the input (`output = activation(dot_product(input, weights) + bias)`) followed by an optional activation function.
      - **`units`**: The number of neurons (or units) in the layer.
      - **`activation`**: The activation function to apply (e.g., `'relu'`, `'softmax'`).
  - **Why Used:** After feature extraction by CNN layers, `Dense` layers are used for classification. They learn non-linear combinations of the extracted features to make final predictions. The final `Dense` layer has `units` equal to the number of output classes.

### Activation Functions

**Activation functions** introduce non-linearity into the neural network. Without them, a neural network would simply be a linear model, no matter how many layers it had, and would not be able to learn complex patterns. An activation function determines whether a neuron should be "activated" or fired, based on its input.

#### ReLU (Rectified Linear Unit)

  - **Full Definition:** $f(x) = \\max(0, x)$
  - **Description:** If the input to the neuron is positive, it outputs the input directly. If the input is zero or negative, it outputs zero.
  - **Graphical Representation:**
    ```
    ^ y
    |
    |      /
    |     /
    |    /
    |   /
    +---/-----> x
    |
    ```
  - **Why Used:**
      - **Computationally Efficient:** Simple to compute.
      - **Mitigates Vanishing Gradients:** Unlike sigmoid or tanh, ReLU does not saturate for positive inputs, which helps alleviate the vanishing gradient problem, allowing deeper networks to train more effectively.
      - **Sparsity:** Introduces sparsity by outputting zero for negative inputs, which can be beneficial for network capacity.
  - **Drawback:** Can suffer from the "dying ReLU" problem where neurons get stuck returning zero if their input is always negative.

#### Softmax

  - **Full Definition:** For an input vector $z = [z\_1, z\_2, ..., z\_K]$, the softmax function outputs a probability distribution $P$ where $P\_i = \\frac{e^{z\_i}}{\\sum\_{j=1}^{K} e^{z\_j}}$.
  - **Description:** Converts a vector of arbitrary real values into a probability distribution, where each value is between 0 and 1, and all values sum to 1.
  - **Why Used:**
      - **Multi-Class Classification Output:** Ideal for the output layer of a multi-class classification problem where each input belongs to exactly one class (e.g., identifying a digit from 0 to 9).
      - **Probabilistic Interpretation:** The output values can be interpreted as the probability of the input belonging to each class.
  - **Usage in Code:** Used in the final `Dense` layer to get class probabilities for the 10 digits.

#### Other Common Activation Functions

  - **Sigmoid ($f(x) = \\frac{1}{1 + e^{-x}}$):** Squashes values between 0 and 1. Historically used in output layers for binary classification, but suffers from vanishing gradients for deep networks.
  - **Tanh ($f(x) = \\frac{e^x - e^{-x}}{e^x + e^{-x}}$):** Squashes values between -1 and 1. Often performs better than sigmoid but still susceptible to vanishing gradients.
  - **Leaky ReLU:** A variation of ReLU that allows a small, non-zero gradient when the input is negative, addressing the "dying ReLU" problem.
  - **ELU (Exponential Linear Unit):** Similar to ReLU but with a smooth, negative curve for negative inputs, helping to alleviate the dying ReLU problem and potentially leading to faster learning.

### Loss Functions

A **loss function** (or cost function) quantifies the difference between the predicted output of a model and the true target value. During training, the goal of the optimizer is to minimize this loss function.

#### Categorical Crossentropy

  - **Full Definition:** For a true one-hot encoded label $y$ and a predicted probability distribution $\\hat{y}$, the loss is $L = -\\sum\_{i=1}^{C} y\_i \\log(\\hat{y}\_i)$, where $C$ is the number of classes.
  - **Description:** This loss function is commonly used for multi-class classification problems where the true labels are provided in a one-hot encoded format (as done with `to_categorical`). It measures the performance of a classification model whose output is a probability value between 0 and 1. It increases as the predicted probability diverges from the actual label.
  - **Why Used:** It's the standard choice for multi-class classification when dealing with one-hot encoded labels and a `softmax` activation in the output layer. It penalizes predictions that are confident but wrong heavily.

#### Other Common Loss Functions

  - **Binary Crossentropy:** Used for binary classification (two classes).
  - **Mean Squared Error (MSE):** Used for regression problems (predicting continuous values). Measures the average of the squares of the errors.
  - **Sparse Categorical Crossentropy:** Similar to categorical crossentropy but used when true labels are integers (0, 1, 2, ...) rather than one-hot encoded.

### Optimizers

An **optimizer** is an algorithm used to modify the attributes of the neural network, such as weights and learning rate, to reduce the loss function. It essentially guides the network in the right direction (down the "loss landscape") during training.

#### Adam Optimizer

  - **Full Definition:** Adaptive Moment Estimation. It's an adaptive learning rate optimization algorithm.
  - **Description:** Adam combines the advantages of two other extensions of stochastic gradient descent:
    1.  **AdaGrad:** Which works well with sparse gradients.
    2.  **RMSProp:** Which works well in online and non-stationary settings.
        It calculates adaptive learning rates for each parameter, providing both a bias correction and exponentially decaying averages of past gradients and squared gradients.
  - **Why Used:**
      - **Generally Recommended:** Often the default choice for many deep learning tasks due to its good performance across a wide range of problems.
      - **Efficient:** Requires less memory and is computationally efficient.
      - **Adaptive Learning Rate:** Adjusts the learning rate for each parameter, leading to faster convergence and better performance.

#### Other Common Optimizers

  - **SGD (Stochastic Gradient Descent):** The simplest optimizer, which updates weights using the gradient of the loss function calculated on a single training example or a small batch.
  - **RMSprop:** An adaptive learning rate optimizer that divides the learning rate by an exponentially decaying average of squared gradients.
  - **Adagrad:** Adapts the learning rate to the parameters, performing larger updates for infrequent parameters and smaller updates for frequent parameters.
  - **Adadelta:** An extension of Adagrad that aims to reduce its aggressive, monotonically decreasing learning rate.

### Metrics

**Metrics** are used to quantify the performance of a model. Unlike loss functions (which are used for optimizing the model), metrics are typically human-interpretable and provide insights into how well the model is solving the problem.

#### Accuracy

  - **Full Definition:** `Accuracy = (Number of Correct Predictions) / (Total Number of Predictions)`
  - **Description:** Measures the proportion of correctly classified samples. In a multi-class classification problem, it calculates how often the model's prediction matches the true label.
  - **Why Used:** It's a straightforward and widely understood metric for classification tasks, indicating the overall correctness of the model.

### Epochs

  - **Definition:** One **epoch** means that the entire training dataset has been passed forward and backward through the neural network exactly once.
  - **Impact:**
      - Too few epochs: The model might be "underfit" (hasn't learned enough from the data).
      - Too many epochs: The model might "overfit" (starts to memorize the training data rather than learning general patterns, leading to poor performance on unseen data).
  - **Usage in Code:** `epochs=5` means the model will iterate through the entire training dataset 5 times.

### Batch Size

  - **Definition:** The **batch size** is the number of training examples utilized in one iteration. The model's weights are updated after processing each batch.
  - **Impact:**
      - **Large Batch Size:**
          - Pros: Stable gradient estimates, faster training time per epoch.
          - Cons: May get stuck in local minima, requires more memory, less generalization sometimes.
      - **Small Batch Size:**
          - Pros: More noisy gradient (can help escape local minima), better generalization, less memory.
          - Cons: Slower training time per epoch, more noisy updates.
  - **Usage in Code:** `batch_size=64` means the model processes 64 images at a time before updating its weights.

### One-Hot Encoding

  - **Definition:** A process by which categorical variables are converted into a form that could be provided to ML algorithms to do a better job in prediction. For integer labels, it transforms a single integer into a binary vector where only one bit is 'hot' (1) and the rest are 'cold' (0).
  - **Example:** For 10 classes (0-9):
      - Label `3` becomes `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`
      - Label `7` becomes `[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]`
  - **Why Used:**
      - Many machine learning algorithms cannot work directly with categorical data.
      - It prevents the algorithm from assuming an ordinal relationship between categories (e.g., that 3 is "greater" than 2).
      - It's a requirement for `categorical_crossentropy` loss.

### Normalization

  - **Definition:** The process of scaling numerical data to a standard range (e.g., 0 to 1 or -1 to 1). For image pixel values, this typically means dividing by the maximum pixel value (255 for 8-bit images).
  - **Why Used:**
      - **Faster Convergence:** Normalizing inputs helps the optimization algorithm converge more quickly. If features have vastly different scales, the optimization landscape can be elongated and difficult to navigate.
      - **Prevent Gradient Issues:** Large input values can lead to large gradients, potentially causing exploding gradients or making the training process unstable.
      - **Improved Performance:** Many activation functions and optimizers perform better with normalized inputs.

<!-- end list -->

```
```
