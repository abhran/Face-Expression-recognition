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
from keras.layers import BatchNormalization, Activation, ZeroPadding2D,MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import CSVLogger

from keras.optimizers import Adam
from keras.models import load_model


# no=100

# model=load_model(f"saved_model/senti_save_model{no}.h5")


# K.tensorflow_backend.set_session(sess)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)
# sess = tf.Session(config=config)
# set_session(sess)  # set this TensorFlow session as the default session for Keras

tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))










# def train(x):
def train(x):
    # first input model
    # visible = Input(shape=input_shape, name='input')
    num_classes = 7
    #the 1-st block
    conv1_1 = Conv2D(64, kernel_size=3, activation='relu', padding='same', name = 'conv1_1')(x)
    conv1_1 = BatchNormalization()(conv1_1)
    conv1_2 = Conv2D(64, kernel_size=3, activation='relu', padding='same', name = 'conv1_2')(conv1_1)
    conv1_2 = BatchNormalization()(conv1_2)
    pool1_1 = MaxPooling2D(pool_size=(2,2), name = 'pool1_1')(conv1_2)
    drop1_1 = Dropout(0.3, name = 'drop1_1')(pool1_1)

    #the 2-nd block
    conv2_1 = Conv2D(128, kernel_size=3, activation='relu', padding='same', name = 'conv2_1')(drop1_1)
    conv2_1 = BatchNormalization()(conv2_1)
    conv2_2 = Conv2D(128, kernel_size=3, activation='relu', padding='same', name = 'conv2_2')(conv2_1)
    conv2_2 = BatchNormalization()(conv2_2)
    conv2_3 = Conv2D(128, kernel_size=3, activation='relu', padding='same', name = 'conv2_3')(conv2_2)
    conv2_2 = BatchNormalization()(conv2_3)
    pool2_1 = MaxPooling2D(pool_size=(2,2), name = 'pool2_1')(conv2_3)
    drop2_1 = Dropout(0.3, name = 'drop2_1')(pool2_1)

    #the 3-rd block
    conv3_1 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv3_1')(drop2_1)
    conv3_1 = BatchNormalization()(conv3_1)
    conv3_2 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv3_2')(conv3_1)
    conv3_2 = BatchNormalization()(conv3_2)
    conv3_3 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv3_3')(conv3_2)
    conv3_3 = BatchNormalization()(conv3_3)
    conv3_4 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv3_4')(conv3_3)
    conv3_4 = BatchNormalization()(conv3_4)
    pool3_1 = MaxPooling2D(pool_size=(2,2), name = 'pool3_1')(conv3_4)
    drop3_1 = Dropout(0.3, name = 'drop3_1')(pool3_1)

    #the 4-th block
    conv4_1 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv4_1')(drop3_1)
    conv4_1 = BatchNormalization()(conv4_1)
    conv4_2 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv4_2')(conv4_1)
    conv4_2 = BatchNormalization()(conv4_2)
    conv4_3 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv4_3')(conv4_2)
    conv4_3 = BatchNormalization()(conv4_3)
    conv4_4 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv4_4')(conv4_3)
    conv4_4 = BatchNormalization()(conv4_4)
    pool4_1 = MaxPooling2D(pool_size=(2,2), name = 'pool4_1')(conv4_4)
    drop4_1 = Dropout(0.3, name = 'drop4_1')(pool4_1)

    #the 5-th block
    conv5_1 = Conv2D(512, kernel_size=3, activation='relu', padding='same', name = 'conv5_1')(drop4_1)
    conv5_1 = BatchNormalization()(conv5_1)
    conv5_2 = Conv2D(512, kernel_size=3, activation='relu', padding='same', name = 'conv5_2')(conv5_1)
    conv5_2 = BatchNormalization()(conv5_2)
    conv5_3 = Conv2D(512, kernel_size=3, activation='relu', padding='same', name = 'conv5_3')(conv5_2)
    conv5_3 = BatchNormalization()(conv5_3)
    conv5_4 = Conv2D(512, kernel_size=3, activation='relu', padding='same', name = 'conv5_4')(conv5_3)
    conv5_3 = BatchNormalization()(conv5_3)
    pool5_1 = MaxPooling2D(pool_size=(2,2), name = 'pool5_1')(conv5_4)
    drop5_1 = Dropout(0.3, name = 'drop5_1')(pool5_1)

    #Flatten and output
    flatten = Flatten(name = 'flatten')(drop5_1)
    output = Dense(num_classes, activation='softmax', name = 'output')(flatten)
    return output





inp = keras.Input(shape=(48,48,1))
x=train(inp)
model=keras.Model(inp,x)






# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=4e-4,
#     decay_steps=500000,
#     decay_rate=0.9,
#     staircase=True)

opt = Adam(lr=0.0005, decay=0.0005 / 10)
# opt = Adam(lr=lr_schedule, decay=1e-6)

# print("[INFO] compiling model...")
# pixel_cnn.compile(optimizer=opt, loss=losses, loss_weights=lossWeights)#, metrics=["accuracy"])

# opt=keras.optimizers.RMSprop(learning_rate=lr_schedule,decay=0.95,momentum=0.9, epsilon=1e-8, name="RMSprop")


loss = tf.keras.losses.categorical_crossentropy
model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
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


        print(f"Epoch : {j}/{1000}     ",end="" )
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




        



    if j%10==0:

        hist.to_csv(f"val_rerun/history{j}.csv")
        model.save(f'saved_model_rerun/senti_saved_model{j}.h5')











