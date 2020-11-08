import tensorflow as tf 
import keras
import numpy as np 
from keras.models import load_model
from PIL import Image
import time
import os
from tqdm import tqdm
from os import path
from tqdm import trange
import pandas as pd
from ast import literal_eval
# import keras







# def explode_inrows(data,coln,rowv,colv,file):
#     dt=data[coln].apply(literal_eval)
#     tr=dt.apply(np.array)
#     tx=np.array(tr)
#     abh=np.concatenate(tx)
#     abh=abh.reshape(len(rowv),len(colv))
#     df = pd.DataFrame(data = abh,  
#                   index = rowv,  
#                   columns = colv) 
#     df.to_csv(f'{file}.csv')
#     return df



# data=pd.read_csv('dt.csv')
# datax=explode_inrows(data.iloc[32000 :  ] , 'pixels' ,[i for i in range(0,3887)],[f'{i}' for i in range(0,48*48)] ,"filexxx")






datax=pd.read_csv('file1.csv').astype(int)
data=pd.read_csv('Train.csv')
datay=data['emotion'].astype(int)
y=datay.iloc[4000:8000]
x=np.array(datax)
x=x[:,1:]
x_=x.reshape(4000,48,48,1)
y=np.array(y)
x=x_/127.5-1
print("Data loaded..................................................................................................")
model=load_model("model/senti_saved_model600.h5")
print("Model Loaded.................................................................................................")
y_=model(x)
y_=np.argmax(y_,axis=1)
a=y_==y
a=list(a)
print(" Accuracy on the dataset is = " ,(sum(a)/len(a))*100, " %")