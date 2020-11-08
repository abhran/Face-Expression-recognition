import os
import tqdm
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval


def get_training_data(self,datafolder,image_width,image_height,channels):
    print("Loading training data...")

    training_data = []
    #Finds all files in datafolder
    filenames = os.listdir(datafolder)
    for filename in tqdm(filenames):
        #Combines folder name and file name.
        path = os.path.join(datafolder,filename)
        #Opens an image as an Image object.
        image = Image.open(path)
        #Resizes to a desired size.
        image = image.resize((self.image_width,self.image_height))
        #Creates an array of pixel values from the image.
        pixel_array = np.asarray(image)
        training_data.append(pixel_array)

        #training_data is converted to a numpy array
    training_data = np.reshape(training_data,(-1,image_width,image_height,channels))
    return training_data

emotion = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3,
           'Sad': 4, 'Surprise': 5, 'Neutral': 6}
emo     = ['Angry', 'Disgust','Fear', 'Happy',
           'Sad', 'Surprise', 'Neutral']

data=pd.read_csv('Train.csv')
print(data.describe())
# print(train)

# def plot_distribution(y1, y2, data_names, ylims =[1000,1000]): 
#     """
#     The function is used to plot the distribution of the labels of provided dataset 
#     """
#     # colorset = brewer2mpl.get_map('Set3', 'qualitative', 6).mpl_colors
#     fig = plt.figure(figsize=(8,4))
#     ax1 = fig.add_subplot(1,2,1)
#     ax1.bar(np.arange(1,7), np.bincount(y1), alpha=0.8)
#     ax1.set_xticks(np.arange(1.25,7.25,1))
#     ax1.set_xticklabels( rotation=60, fontsize=14)
#     ax1.set_xlim([0, 8])
#     ax1.set_ylim([0, ylims[0]])
#     ax1.set_title(data_names[0])
#     ax2 = fig.add_subplot(1,2,2)
#     ax2.bar(np.arange(1,7), np.bincount(y2), alpha=0.8)
#     ax2.set_xticks(np.arange(1.25,7.24,1))
#     ax2.set_xticklabels( rotation=60, fontsize=14)
#     ax2.set_xlim([0, 8])
#     ax2.set_ylim([0, ylims[1]])
#     ax2.set_title(data_names[1])
#     plt.tight_layout()
#     plt.show()
    
# plot_distribution(y_train_labels, y_public_labels, \
#                   ['Train dataset', 'Public dataset'], \
#                   ylims =[8000,1000]) 

ts=data.drop(['Usage'],axis=1)
trainx=ts['pixels']
print(np.array(trainx))
def fun(X):
    pixel=[]
    for i in X.split():
        pixel.append(int(i))
    return pixel


trx=trainx.apply(fun)


trx.to_csv('dts.csv')
