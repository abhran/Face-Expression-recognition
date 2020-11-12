import streamlit as st
import matplotlib.pyplot as plt
file_bytes = st.file_uploader("Upload a file", type=("png", "jpg"))
b=plt.imread(file_bytes)
b=plt.imread(file_bytes)
# st.write(b)
# plt.imshow(b)
# st.pyplot(plt)
st.image(b,width=600)
import cv2
# import sys
# import os
import numpy as np
import tensorflow as tf 
import keras
# import numpy as np 
from keras.models import load_model
from PIL import Image
import time
# import os
# from tqdm import tqdm
# from os import path
# from tqdm import trange
# import pandas as pd
# from ast import literal_eval










class FaceCropper(object):
#     cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
#     CASCADE_PATH = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
    CASCADE_PATH='haarcascade_frontalface_default.xml'
    #  = "data/haarcascades/haarcascade_frontalface_default.xml"

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(self.CASCADE_PATH)

    def generate(self, image, show_result):
        k=[]

        # img = cv2.imread(image)
        # if (img is None):
        #     print("Can't open image file")
        #     return 0

        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(image, 1.1, 3, minSize=(48, 48))
        if (faces is None):
            print('Failed to detect face')
            return 0

        # if (show_result):
        #     for (x, y, w, h) in faces:
        #         cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        #     cv2.imshow('img', img)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

        facecnt = len(faces)
        print("Detected faces: %d" % facecnt)
        i = 0
        height, width = image.shape[:2]

        for (x, y, w, h) in faces:
            r = max(w, h) / 2
            centerx = x + w / 2
            centery = y + h / 2
            nx = int(centerx - r)
            ny = int(centery - r)
            nr = int(r * 2)

            faceimg = image[ny:ny+nr, nx:nx+nr]
            lastimg = cv2.resize(faceimg, (48, 48))
            i += 1
            # cv2.imwrite("image%d.jpg" % i, lastimg)
            k.append(lastimg)
            
        return k

@st.cache   
def pred(no):
    return load_model(f"saved_model/senti_saved_model{no}.h5")



def arrange(arr1):
  m=[]
  k=list(arr1)
  p=k.copy()
  k.sort(reverse=True)
  for i in range(0,len(k)):
    # print(k)
    # print(p)
    aa=p.index(k[i])
    m.append(aa)
  return m



detecter = FaceCropper()
a=detecter.generate(b, True)
# st.write(a)
l=len(a)
emo     = ['Angry','Disgust', 'Fear', 'Happy',
           'Sad', 'Surprise', 'Neutral']

for i in range(0,l):
    st.image(a[i], width=200)
    k=np.dot(a[i], [.3, .6, .1]) 

    k=np.reshape(k,(48,48,1))
    k=k/255
    kp=k-1
    kp=np.reshape(kp,(1,48,48,1))
    model=pred(900)
    prediction=model.predict(kp)
    prediction=np.reshape(prediction,(7,))


    emonolist=arrange(prediction)

    for emono in emonolist:
        st.write(emo[emono])




    
