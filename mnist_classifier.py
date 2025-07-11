# import libraries 
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical


# download dataset
# (train_img, train_label), (test_img, test_label) = datasets.mnist.load_data()


# # Save to a compressed .npz file
# np.savez('dataset/mnist.npz', 
#          train_img=train_img, 
#          train_label=train_label, 
#          test_img=test_img, 
#          test_label=test_label)


# Load from the local .npz file
path = 'dataset/mnist.npz'

with np.load(path, allow_pickle=True) as data:
    train_img = data['x_train']
    train_label = data['y_train']
    test_img = data['x_test']
    test_label = data['y_test']


# Preprocessing ( Normalize the pixel value to be between 0 , 1 )
train_img = train_img / 250.0
test_img = test_img / 250.0

print(train_img.shape) # (60000, 28, 28)
print(test_img.shape) # (10000, 28, 28)

#‌ Reshape images to (28,28,1)
train_img = train_img.reshape((train_img.shape[0], 28, 28, 1))
test_img = test_img.reshape((test_img.shape[0], 28, 28, 1))

print(train_img.shape) # (60000, 28, 28, 1)
print(test_img.shape) # (10000, 28, 28, 1)

#‌ Convert label to one-hot encoding
train_label = to_categorical(test_label)
test_label = to_categorical(test_label)

# Make Model CNN
model = models.Sequential()

#‌ Add layers in model
#‌# Input layer
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))

#‌# hiden layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))

#‌# Output layer (10 mode => (0,1,..,8,9))
model.add(layers.Dense(10, activation='softmax'))

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(train_img, train_label, epochs=5, batch_size=64, validation_data=(test_img, test_label))

# Test Model
loss_, acc_ = model.evaluate(test_img, test_label)
print(f'Test Accuracy: {acc_*100:.3}% \t Test Loss: {loss_}')



