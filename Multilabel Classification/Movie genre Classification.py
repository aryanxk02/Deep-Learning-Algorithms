from tensorflow.keras.preprocessing import image
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras import Sequential
from tensorflow.keras.utils import load_img, img_to_array
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import sys

import tensorflow.keras
import pandas as pd
import sklearn as sk
import tensorflow as tf

print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {tensorflow.keras.__version__}")
print()
print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-Learn {sk.__version__}")
gpu = len(tf.config.list_physical_devices('GPU'))>0
print("GPU is", "available" if gpu else "NOT AVAILABLE")

image_directory = "merge_folder"

df = pd.read_csv('movie-metadata.csv')
df.head()

df.columns

df.shape

SIZE = 250
X = []

for i in (range(df.shape[0])):
    img = tf.keras.utils.load_img(image_directory+'/'+df['Id'][i]+'.jpg', target_size=(SIZE, SIZE))
    img = image.img_to_array(img)
    img = img/255
    X.append(img)
    
X = np.array(X)

y = np.array(df.drop(['Id', 'Genre'], axis=1))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=20, test_size=0.3)


import keras
import keras.layers as layers

# VGG
# model = keras.Sequential()

# model.add(layers.Conv2D(filters=16, kernel_size=(5, 5), 
#                         activation="relu", 
#                         input_shape=(250, 250, 3)))
# model.add(layers.BatchNormalization())
# model.add(layers.MaxPool2D(pool_size=(3, 3)))
# model.add(Dropout(0.2))

# model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), 
#                         activation="relu", 
#                         ))
# model.add(layers.BatchNormalization())
# model.add(layers.MaxPool2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))

# model.add(layers.Conv2D(filters=64, kernel_size=(5, 5), 
#                         activation="relu"
#                         ))
# model.add(layers.BatchNormalization())
# model.add(layers.MaxPool2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))

# model.add(layers.Conv2D(filters=64, kernel_size=(5, 5), 
#                         activation="relu"
#                         ))
# model.add(layers.BatchNormalization())
# model.add(layers.MaxPool2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))

# model.add(layers.Flatten())
# model.add(layers.Dense(128, activation="relu"))
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(64, activation="relu"))
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(25, activation="sigmoid"))


# AlexNet

model = keras.Sequential()

model.add(layers.Conv2D(filters=96, kernel_size=(11, 11), 
                        strides=(4, 4), activation="relu", 
                        input_shape=(250, 250, 3)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D(pool_size=(3, 3), strides= (2, 2)))

model.add(layers.Conv2D(filters=256, kernel_size=(5, 5), 
                        strides=(1, 1), activation="relu", 
                        padding="same"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))

model.add(layers.Conv2D(filters=384, kernel_size=(3, 3), 
                        strides=(1, 1), activation="relu", 
                        padding="same"))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(filters=384, kernel_size=(3, 3), 
                        strides=(1, 1), activation="relu", 
                        padding="same"))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), 
                        strides=(1, 1), activation="relu", 
                        padding="same"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(4096, activation="relu"))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(25, activation="softmax"))

model.compile(loss='sparse_categorical_crossentropy', 
              optimizer=tf.optimizers.SGD(lr=0.001), 
              metrics=['accuracy'])

model.summary()

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
