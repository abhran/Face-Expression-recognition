import os
import tqdm
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval
# import keras



def intt(x):
    return [int(i) for i in x]


# col=[f'{i}' for i in range(0,48*48)]
# row=[i for i in range(0,10)]
data=pd.read_csv('dt.csv')
# x=data.head(10)
# xy=x['pixels'].apply(literal_eval)
# xy['pp']=xy.apply(intt)

# tr=xy['pp'].apply(np.array)

# tx=np.array(tr)
# abh=np.concatenate(tx)
# abh=abh.reshape(10,2304)
# df = pd.DataFrame(data = abh,  
#                   index = row,  
#                   columns = col) 
# print(pd.DataFrame(abh))
# print(abh.reshape(10,2304))
# print((xxxx))
# print(df)

def explode_inrows(data,coln,rowv,colv,file):
    dt=data[coln].apply(literal_eval)
    tr=dt.apply(np.array)
    tx=np.array(tr)
    abh=np.concatenate(tx)
    abh=abh.reshape(len(rowv),len(colv))
    df = pd.DataFrame(data = abh,  
                  index = rowv,  
                  columns = colv) 
    df.to_csv(f'{file}.csv')
    # return df
for i in range (0,8):
    explode_inrows(data.iloc[(4000*i) : 4000*(i+1) ] , 'pixels' ,[i for i in range(0,4000)],[f'{i}' for i in range(0,48*48)] , f"file{i}")

# dt=pd.read_csv('file.csv')
# print(dt)   








