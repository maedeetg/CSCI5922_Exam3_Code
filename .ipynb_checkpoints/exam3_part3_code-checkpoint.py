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

def one_hot_encode(y, num_labels):
    '''Create one-hot encoding for numerical labels. If the labels
       are categorical, they need to be changed to numerical first.
       This should be done when reading in the data.'''
    for i in range(len(y)):
        label = y[i]
        
        if label == "football":
            y[i] = 0
            
        elif label == "politics":
            y[i] = 1
            
        else:
            y[i] = 2
            
    n = len(y)
    one_hot_labels = np.zeros((n, num_labels))

    for i in range(n):
        one_hot_labels[i, int(y[i][0]) - 1] = 1

    return(one_hot_labels)

##########################################################
# Import data and split into training and testing
data = pd.read_csv("Final_News_DF_Labeled_ExamDataset.csv") # (1493, 301)

# 80% of data will be training, use seed = 42
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42) # (1194, 301)

x_train = train_data.loc[:, data.columns != "LABEL"].to_numpy() # (1194, 300)
x_test = test_data.loc[:, data.columns != "LABEL"].to_numpy() # (299, 300)
y_train = train_data.loc[:, data.columns == "LABEL"].to_numpy() # (1194, 300)
y_test = test_data.loc[:, data.columns == "LABEL"].to_numpy() # (299, 300)

# # We need to one-hot-encode the labe
# y_train_one_hot = one_hot_encode(y_train, 3)
# y_test_one_hot = one_hot_encode(y_test, 3)

# What does the data look like?
print("The first value of x_train is: \n", x_train[0])
print("The shape of x_train is: ", x_train.shape)

print("The first value of y_train is: ", y_train[0])
print("The shape of y_train is: ", y_train.shape)

print("The first value of x_test is: \n", x_test[0])
print("The shape of x_test is: ", x_test.shape)

print("The first value of y_test is: ", y_test[0])
print("The shape of y_test is: ", y_test.shape)

# We need to one-hot-encode the labels
y_train_one_hot = one_hot_encode(y_train, 3)
y_test_one_hot = one_hot_encode(y_test, 3)

# Need to change datatypes so they are compataible for keras later
x_train = np.array(x_train, dtype=np.float32) 
y_train = np.array(y_train_one_hot, dtype=np.float32) 
x_test = np.array(x_test, dtype=np.float32)
y_test = np.array(y_test_one_hot, dtype=np.float32)

##########################################################
# # Create sequential ANN. 
# # BUILD MODEL
# ANN = keras.Sequential([tf.keras.layers.Flatten(input_shape=(300, )),
#                                 tf.keras.layers.Dropout(0.5),
#                                 tf.keras.layers.Dense(100, activation='relu'),
#                                 tf.keras.layers.Dense(3, activation='softmax')])

# # MODEL SUMMARY
# ANN.summary()

# # COMPILE MODEL
# ANN.compile(loss="categorical_crossentropy",
#             metrics=["accuracy"],
#             optimizer='adam')

# # FIT THE MODEL TO TRAINING DATA
# Fit = ANN.fit(x_train, y_train, epochs = 50, validation_data = (x_test, y_test))

# # PLOT RESULTS
# # Accuracy
# plt.plot(Fit.history['accuracy'], label = 'training accuracy', color = 'magenta')
# plt.plot(Fit.history['val_accuracy'], label = 'validation accuracy', color = 'purple')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title("Accuracy over Epochs")
# plt.ylim([0.5, 1])
# plt.legend(loc='lower right')
# plt.show()
# # Loss
# plt.plot(Fit.history['loss'], label = 'training loss', color = 'magenta')
# plt.plot(Fit.history['val_loss'], label = 'validation loss', color = 'purple')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title("Loss over Epochs")
# plt.legend(loc='lower right')
# plt.show()

# # TEST
# Test_Loss, Test_Accuracy = ANN.evaluate(x_test, y_test)

# # PREDICT & CONFUSION MATRIX
# predictions = ANN.predict([x_test])
# Max_Values = np.squeeze(np.array(predictions.argmax(axis=1))) # all our label predictions

# y_numeric = np.argmax(y_test, axis = 1)     
# y_hat_numeric = Max_Values
# labels = ['football', 'politics', 'science']
# cm = confusion_matrix(y_numeric, y_hat_numeric)
# ax = plt.subplot()
# sns.heatmap(cm, annot = True, fmt = 'g', ax = ax, cmap = 'flare')  
# ax.set_xlabel("Predicted labels")
# ax.set_ylabel("True labels")
# ax.set_title("Confusion Matrix")
# ax.xaxis.set_ticklabels(labels)
# ax.yaxis.set_ticklabels(labels)
# plt.show()

##########################################################
# # Create sequential CNN
# # Need to reshape the x_train and x_test to be used for a CNN. 
# # I will reshape as (30, 10) instead of the (300,1)
# x_train_reshape = x_train.reshape(1194, 30, 10)
# x_test_reshape = x_test.reshape(299, 30, 10)

# # BUILD MODEL
# CNN = keras.Sequential([
#     tf.keras.layers.Conv2D(input_shape = (30, 10, 1), kernel_size = (3, 3), filters = 32, activation = 'relu'), 
#     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#     tf.keras.layers.Conv2D(kernel_size = (2, 2), filters = 64, activation = 'relu'),
#     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(3, activation = "softmax")])

# CNN.summary()

# # COMPILE MODEL
# CNN.compile(loss="categorical_crossentropy",
#             metrics=["accuracy"],
#             optimizer='adam')


# # FIT THE MODEL TO TRAINING DATA
# Fit = CNN.fit(x_train_reshape, y_train, epochs = 50, validation_data = (x_test_reshape, y_test))

# # PLOT RESULTS
# # Accuracy
# plt.plot(Fit.history['accuracy'], label = 'training accuracy', color = 'magenta')
# plt.plot(Fit.history['val_accuracy'], label = 'validation accuracy', color = 'purple')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title("Accuracy over Epochs")
# plt.ylim([0.5, 1])
# plt.legend(loc='lower right')
# plt.show()
# # Loss
# plt.plot(Fit.history['loss'], label = 'training loss', color = 'magenta')
# plt.plot(Fit.history['val_loss'], label = 'validation loss', color = 'purple')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title("Loss over Epochs")
# plt.legend(loc='lower right')
# plt.show()

# # TEST
# Test_Loss, Test_Accuracy = CNN.evaluate(x_test_reshape, y_test)

# # PREDICT & CONFUSION MATRIX
# predictions = CNN.predict([x_test_reshape])
# Max_Values = np.squeeze(np.array(predictions.argmax(axis=1))) # all our label predictions

# y_numeric = np.argmax(y_test, axis = 1)     
# y_hat_numeric = Max_Values
# labels = ['football', 'politics', 'science']
# cm = confusion_matrix(y_numeric, y_hat_numeric)
# ax = plt.subplot()
# sns.heatmap(cm, annot = True, fmt = 'g', ax = ax, cmap = 'flare')  
# ax.set_xlabel("Predicted labels")
# ax.set_ylabel("True labels")
# ax.set_title("Confusion Matrix")
# ax.xaxis.set_ticklabels(labels)
# ax.yaxis.set_ticklabels(labels)
# plt.show()

##########################################################
# Create sequential LSTM
# BUILD MODEL
LSTM = tf.keras.Sequential([
    tf.keras.layers.LSTM(units = 100, input_shape =(300, 1)), 
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(50, activation='tanh'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(3, activation = 'softmax')])

LSTM.summary()

LSTM.compile(loss = keras.losses.CategoricalCrossentropy(from_logits = False), metrics = ["accuracy"], optimizer = "adam")

Fit = LSTM.fit(x_train, y_train, epochs = 50, validation_data =(x_test, y_test))

# PLOT RESULTS
# Accuracy
plt.plot(Fit.history['accuracy'], label = 'training accuracy', color = 'magenta')
plt.plot(Fit.history['val_accuracy'], label = 'validation accuracy', color = 'purple')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title("Accuracy over Epochs")
plt.legend(loc='lower right')
plt.show()
# Loss
plt.plot(Fit.history['loss'], label = 'training loss', color = 'magenta')
plt.plot(Fit.history['val_loss'], label = 'validation loss', color = 'purple')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title("Loss over Epochs")
plt.legend(loc='lower right')
plt.show()

# TEST
test_loss, test_accuracy = LSTM.evaluate(x_test, y_test)

# PREDICT & CONFUSION MATRIX
predictions = LSTM.predict([x_test])
Max_Values = np.squeeze(np.array(predictions.argmax(axis=1))) # all our label predictions

y_numeric = np.argmax(y_test, axis = 1)     
y_hat_numeric = Max_Values
labels = ['football', 'politics', 'science']
cm = confusion_matrix(y_numeric, y_hat_numeric)
ax = plt.subplot()
sns.heatmap(cm, annot = True, fmt = 'g', ax = ax, cmap = 'flare')  
ax.set_xlabel("Predicted labels")
ax.set_ylabel("True labels")
ax.set_title("Confusion Matrix")
ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels)
plt.show()
##########################################################