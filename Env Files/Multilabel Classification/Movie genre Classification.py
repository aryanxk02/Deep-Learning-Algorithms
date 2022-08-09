# import libraries
from keras.preprocessing import image
from keras.model import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization

from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# set dataset path
image_directory = 'Users/aryan/Desktop/DL/movie_data'

df = pd.read_csv('movie_metadata.csv')
df.shape

SIZE = 250
X = []

for i in tqdm(range(df.shape[0])):
    img = image.load_image(image_directory+df['id'][i]+'.jpg', target_size=(SIZE, SIZE))
    img = image.img_to_array(img)
    img = img/255
    X.append(img)
    
X = np.array(x)
