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