
import tensorflow as tf;
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input # type: ignore
from tensorflow.keras.layers import GlobalMaxPooling2D # type: ignore
import cv2
import numpy as np
from numpy.linalg import norm
import pickle
from sklearn.neighbors import NearestNeighbors

feature_list=np.array(pickle.load(open("featurevector.pkl","rb")))
filename=pickle.load(open("filename.pkl","rb"))
model=ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable=False

import tensorflow as tf
from tensorflow.keras.applications import VGG16 # type: ignore
from tensorflow.keras.layers import GlobalMaxPool2D, Input # type: ignore
from tensorflow.keras.models import Model # type: ignore


input_shape = (224, 224, 3)
input_tensor = Input(shape=input_shape)
base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
base_model.trainable = False
x = GlobalMaxPool2D()(base_model.output)
model = Model(inputs=base_model.input, outputs=x)
model.summary()
img=cv2.imread("t2.jpg")
if img is None:
    raise FileNotFoundError("The image file t1.jpg was not found.")
img=cv2.resize(img,(224,224))
img=np.array(img)
expand_img=np.expand_dims(img, axis=0)
pre_img=preprocess_input(expand_img)
result=model.predict(pre_img).flatten()
normalised=result/norm(result)


neighbors=NearestNeighbors(n_neighbors=6,algorithm="brute",metric="euclidean")
neighbors.fit(feature_list)

distance,indices=neighbors.kneighbors([normalised])

for file in indices[0][1:6]:
    imgName = cv2.imread(filename[file])
    if imgName is None:
        print(f"Warning: The image file {filename[file]} was not found or could not be read.")
        continue
    if imgName.size == 0:
        print(f"Warning: The image file {filename[file]} is empty or invalid.")
        continue
    resized_img = cv2.resize(imgName, (640, 480))  # Resize the image before displaying
    cv2.imshow("Frame", resized_img)
    cv2.waitKey(0)
   