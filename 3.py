import os
import tqdm
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval
import tensorflow as tf
from keras.activations import relu
# import keras
import keras
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import CSVLogger



# tensorboard_cb = keras.callbacks.TensorBoard(
#     log_dir='tensorboard',
#     histogram_freq=1,
#     write_graph=True,
#     write_images=True
# )
# import pickle


def train(x):
    x=Conv2D(32, kernel_size=3, strides=2)(x)
    x=LeakyReLU(alpha=0.2)(x)
    x=Dropout(0.25)(x)
    x=Conv2D(64, kernel_size=3, strides=2)(x)
    x=ZeroPadding2D(padding=((0,1),(0,1)))(x)
    x=BatchNormalization(momentum=0.8)(x)
    x=LeakyReLU(alpha=0.2)(x)
    x=Dropout(0.25)(x)
    x=Conv2D(128, kernel_size=3, strides=2)(x)
    x=BatchNormalization(momentum=0.8)(x)
    x=LeakyReLU(alpha=0.2)(x)
    x=Dropout(0.25)(x)
    x=Conv2D(256, kernel_size=3, strides=1)(x)
    x=BatchNormalization(momentum=0.8)(x)
    x=LeakyReLU(alpha=0.2)(x)
    x=Dropout(0.25)(x)
    x=Flatten()(x)
    x=Dense(7, activation='sigmoid')(x)
    return x

inp = keras.Input(shape=(48,48,1))
x=train(inp)
model=keras.Model(inp,x)
loss = tf.keras.losses.categorical_crossentropy
model.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])
print(model.summary())




datax=pd.read_csv('filexxx.csv').astype(int)
data=pd.read_csv('Train.csv')
datay=data['emotion'].astype(int)
y=datay.iloc[32000:]
yval = pd.get_dummies(y, prefix='emotion')
x=np.array(datax)
x=x[:,1:]
x_=x.reshape(3887,48,48,1)
yval=np.array(yval)
xval=x_/127.5-1


datay=data['emotion'].astype(int)


hist=pd.DataFrame({'acc':[],"loss":[],"val_acc":[],"val_loss":[]})

for j in range(1,1001):
    for i in range (0,8):

        hist_={'acc':0,"loss":1,"val_acc":2,"val_loss":3}


        print(f"Epoch : {j}/{400}     ",end="" )
        print(f"Batch : {(i)*125}/{125*8}")
        datax=pd.read_csv(f'file{i}.csv').astype(int)
        y=datay.iloc[i*4000:(i+1)*4000]
        ytest=y
        x=np.array(datax)
        x=x[:,1:]
        x_=x.reshape(4000,48,48,1)
        y = pd.get_dummies(y, prefix='emotion')
        y_=np.array(y)
        x=x_/127.5-1
        # print(x,x.shape)
        # csv_logger = CSVLogger(f'loss_log{i}.csv', append=True, separator=',')
        history_callback=model.fit(x, y_, batch_size=32, epochs=1,validation_data=(xval, yval), shuffle=True, verbose=1)#,callbacks=[tensorboard_cb,csv_logger],verbose=1)


        hist_["acc"] = history_callback.history['accuracy']
        hist_["loss"] = history_callback.history["loss"]
        hist_["val_acc"] = history_callback.history['val_accuracy']
        hist_["val_loss"] = history_callback.history['val_loss']
        history=pd.DataFrame(hist_)
        hist=pd.concat([hist,history])
        print(hist)




        



    if j%20==0:

        hist.to_csv(f"train_val_history/history{j}.csv")
        model.save(f'saved_model/senti_saved_model{j}.h5')


