# Import necessary libraries
import numpy as np
from tensorflow.keras import datasets, layers, models # Specific modules from Keras (TensorFlow's high-level API) for loading datasets, building neural network layers, and creating models.
from tensorflow.keras.utils import to_categorical  # A utility to convert integer labels into one-hot encoded vectors.


# Note: The code had commented-out lines to download and save the MNIST dataset.
#‌ # download dataset
# (train_img, train_label), (test_img, test_label) = datasets.mnist.load_data()

# # Save to a compressed .npz file
# np.savez('dataset/mnist.npz', 
#          x_train=train_img, 
#          y_train=train_label, 
#          x_test=test_img, 
#          y_test=test_label)

# Load the dataset from a local .npz file.
path = 'dataset/mnist.npz'

with np.load(path, allow_pickle=True) as data:
    train_img = data['x_train']
    train_label = data['y_train']
    test_img = data['x_test']
    test_label = data['y_test']


# Preprocessing: Normalize pixel values to be between 0 and 1.
train_img = train_img / 255.0
test_img = test_img / 255.0

print(train_img.shape) # Expected: (60000, 28, 28)
print(test_img.shape)  # Expected: (10000, 28, 28)

# Reshape images to add a channel dimension (for grayscale images, 1 channel).
# Required for Conv2D layers: (batch_size, height, width, channels)
train_img = train_img.reshape((train_img.shape[0], 28, 28, 1))
test_img = test_img.reshape((test_img.shape[0], 28, 28, 1))

print(train_img.shape) # Expected: (60000, 28, 28, 1)
print(test_img.shape)  # Expected: (10000, 28, 28, 1)

# Convert integer labels to one-hot encoding.
# This is crucial for 'categorical_crossentropy' loss.
train_label = to_categorical(train_label)
test_label = to_categorical(test_label) 

# Build the Convolutional Neural Network (CNN) model
model = models.Sequential()

# Add layers to the model
# Input Layer: First Convolutional layer with ReLU activation and MaxPooling
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))

# Hidden Layers: More Convolutional and MaxPooling layers
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Flatten the output from convolutional layers to feed into dense layers
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))

# Output Layer: Dense layer with 10 neurons (for 10 classes, digits 0-9)
# Softmax activation for probability distribution over classes
model.add(layers.Dense(10, activation='softmax'))

# Compile the Model
# Optimizer: Adam (efficient for gradient descent)
# Loss Function: Categorical Crossentropy (for multi-class classification with one-hot labels)
# Metrics: Accuracy (to monitor performance)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the Model
# Fit the model to the training data, validating on the test data after each epoch.
model.fit(train_img, train_label, epochs=5, batch_size=64, validation_data=(test_img, test_label))

# Evaluate the Model's performance on the test set
loss_, acc_ = model.evaluate(test_img, test_label)
print(f'Test Accuracy: {acc_*100:.3}% \t Test Loss: {loss_}')

# Make a prediction for the first image in the test set
predict_ = model.predict(test_img)
print(f'Prediction for first image: {np.argmax(predict_[0])}')
