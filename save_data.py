import os
import cv2
import numpy as np
import mtcnn
import tensorflow as tf
from PIL import Image
physical_devices=tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0],True)
from detect_face import face_extract,load_dataset

print('processing training - data')
x_train, y_train = load_dataset('5-celebrity-faces-dataset/train/',(160,160))

print('processing validation - data')
x_val, y_val = load_dataset('5-celebrity-faces-dataset/val/',(160,160))


with open('data.pkl', 'wb') as f:
    pickle.dump([x_train,y_train,x_val,y_val], f)
print('Data sucessfuly saved')