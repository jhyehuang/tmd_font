# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 09:53:40 2018

@author: huang
"""
import sys
import math
import os
import tensorflow as tf
from  data_folder_split import mkdir ,des_path,load_image
import numpy as np
import pandas as pd

def convert_dataset(list_path, output_dir):
    fd = open(list_path)
    lines=[]
    for line in fd:
        line=line.split(',')
        lines.append(line)
    fd.close()
    output_path = os.path.join(output_dir,
        'data_{}.csv'.format('train'))
    print(output_path)
    batch_size=len(lines)
#    batch_size=50
    #save_arr = pd.DataFrame(save_arr)
    for i in range(batch_size):
        image_data =open(lines[i][1], 'rb')
        image = load_image(image_data)
        image=np.append(image,int(lines[i][2].replace('\n','')))
        if i==0:
            save_arr=image
        else:
            save_arr=np.row_stack((save_arr,image))
        image_data.close()
    save_arr = pd.DataFrame(save_arr) 
    save_arr.to_csv(output_path, decimal=',', encoding='utf-8', index=False, index_label=False)

TF_train_file_dir=des_path+'float_train/'
TF_eval_file_dir=des_path+'float_eval/'
mkdir(TF_train_file_dir)
mkdir(TF_eval_file_dir)
list_train_file_name=des_path+'list_train.txt'
list_eval_file_name=des_path+'list_val.txt'

convert_dataset(list_train_file_name, TF_train_file_dir)
convert_dataset(list_eval_file_name, TF_eval_file_dir)
