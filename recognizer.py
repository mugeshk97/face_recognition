import cv2
import pickle
import matplotlib.pyplot as plt
import numpy as np
import mtcnn
import tensorflow as tf
from PIL import Image
physical_devices=tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0],True)
from detect_face import face_extract

from sklearn.preprocessing import Normalizer
normalizer = Normalizer(norm='l2')

from keras_facenet import FaceNet
embedder = FaceNet()


with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
 
file='5-celebrity-faces-dataset/val/madonna/httpassetsrollingstonecomassetsarticlemadonnadavidbowiechangedthecourseofmylifeforeversmallsquarexmadonnabowiejpg.jpg'
img=plt.imread(file)
face=face_extract(file,(160,160))
face_array = np.asarray(face)
face_array=np.expand_dims(face_array,axis=0)
emded=embedder.embeddings(face_array)
emded=normalizer.fit_transform(emded)
pred=model.predict(emded)
print(pred)
