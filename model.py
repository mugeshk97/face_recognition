import os
import cv2
import numpy as np
import mtcnn
import tensorflow as tf
from PIL import Image
physical_devices=tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0],True)
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import pickle
from keras_facenet import FaceNet
embedder = FaceNet()

with open('data.pkl', 'rb') as f:
    x_train,y_train,x_val,y_val = pickle.load(f)


emb_train_x = embedder.embeddings(x_train)
emb_val_x=embedder.embeddings(x_val)



normalizer = Normalizer(norm='l2')
x_train = normalizer.transform(emb_train_x)
x_val = normalizer.transform(emb_val_x)
encoder = LabelEncoder()
encoder.fit(y_train)
y_train = encoder.transform(y_train)
y_val = encoder.transform(y_val)



model = SVC(kernel='linear', probability=True)
model.fit(x_train,y_train)
# predict

prediction = model.predict(x_val)

# accuracy

print('Accuracy of model ',accuracy_score(y_val, prediction))

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print('Model Saved Sucessfully')