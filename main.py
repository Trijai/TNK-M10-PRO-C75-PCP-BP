import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from keras.models import Sequential,load_model,Model
from keras.layers import Conv2D,MaxPool2D,Dense,Dropout,BatchNormalization,Flatten,Input
from sklearn.model_selection import train_test_split
from cvzone.FaceDetectionModule import FaceDetector

path = "Pneumothorax-New-Dataset"

images = []
categories = []

for img in os.listdir(path):
    try:
        print(img)
    
        type = img.split("_")[0]
        img = cv2.imread(str(path)+"/"+str(img))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(200,200))
        images.append(img)
        categories.append(type)
               
    except:
        print("error in reading")

print("Count of all images", len(images))

# Change the categories to numpy array of int64

# Change images to numpy array


# Split the images and categories using train_test_split


# Create a sequential model


# Add First layer of the model


# Add Second layer of the model


# Add thirds layer of the model              


# Add fourth layer of the model


# Add flatten to model


# Add Dropout of 0.2 to model

# Add dense layer to model with 512 size and relu activation

# Add dense layer to model with 1 size and linear activation and name = 'age'


# Compile the model              

# Print model summary


# tarin the model


# Save the model


# Plot the training and validation accuracy and loss at each epoch


acc = history.history['accuracy']
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()