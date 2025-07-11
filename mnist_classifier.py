# import libraries 
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical


# download dataset
# (train_img, train_label), (test_img, test_label) = datasets.mnist.load_data()


# # Save to a compressed .npz file
# np.savez('mnist_local.npz', 
#          train_img=train_img, 
#          train_label=train_label, 
#          test_img=test_img, 
#          test_label=test_label)


# Load from the local .npz file
data = np.load('mnist_local.npz')
train_img = data['train_img']
train_label = data['train_label']
test_img = data['test_img']
test_label = data['test_label']
