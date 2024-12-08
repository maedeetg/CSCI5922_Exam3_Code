# Import libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow.keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import seaborn as sns

# Create sequential ANN. 
# BUILD MODEL
ANN = keras.Sequential([tf.keras.layers.Flatten(input_shape=(4, )),
                                tf.keras.layers.Dense(4, activation='sigmoid'),
                                tf.keras.layers.Dense(3, activation='relu'),
                                tf.keras.layers.Dense(3, activation='softmax')])

# MODEL SUMMARY
ANN.summary()

# COMPILE MODEL
ANN.compile(loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
            optimizer='adam')

# Create sequential CNN. 
# BUILD MODEL
CNN = keras.Sequential([
    tf.keras.layers.Conv2D(input_shape = (30, 30, 1), kernel_size = (4, 4), filters = 2, activation = 'relu', strides = (1, 1), padding = "same"), 
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(kernel_size = (3, 3), filters = 4, activation = 'relu', strides = (1, 1), padding = "same"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(3, activation = "softmax")])

# MODEL SUMMARY
CNN.summary()

# COMPILE MODEL
CNN.compile(loss="categorical_crossentropy",
            metrics=["accuracy"],
            optimizer='adam')