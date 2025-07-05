# import libraries 
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical


# # load dataset
(train_img, train_label), (test_img, test_label) = datasets.mnist.load_data()
