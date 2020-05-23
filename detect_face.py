import os
import cv2
import numpy as np
import mtcnn
import tensorflow as tf
from PIL import Image
physical_devices=tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0],True)

def face_extract(file, size):
    image = cv2.imread(file)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    pixels = np.asarray(image)
    detector = mtcnn.MTCNN()
    results = detector.detect_faces(pixels)
    
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    
    # extract the face
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(size)
    face_array = np.asarray(image)
    return face_array
	
def load_class(directory,size):
    faces = list()
    for filename in os.listdir(directory):
        path = directory + filename
        face = face_extract(path,size)
        faces.append(face)
    return faces
	
def load_dataset(directory,size):
    X, y = list(), list()
    
    for subdir in os.listdir(directory):
        path = directory + subdir + '/'
        print('Extracting the faces of class--',subdir)
        faces = load_class(path,size)
        labels = [subdir for _ in range(len(faces))]
        print(f'Loaded %d images for the class: %s' % (len(faces), subdir))
        X.extend(faces)
        y.extend(labels)
    return np.asarray(X), np.asarray(y)
    
