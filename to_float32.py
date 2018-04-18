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

def convert_dataset(list_path, output_dir, _NUM_SHARDS=5):
    fd = open(list_path)
    lines=[]
    for line in fd:
        line=line.split(',')
        lines.append(line)
    fd.close()
    num_per_shard = int(math.ceil(len(lines) / float(_NUM_SHARDS)))
    
    for shard_id in range(_NUM_SHARDS):
        output_path = os.path.join(output_dir,
            'data_{:05}-of-{:05}.csv'.format(shard_id, _NUM_SHARDS))
        print(output_path)
        start_ndx = shard_id * num_per_shard
        end_ndx = min((shard_id + 1) * num_per_shard, len(lines))
        save_arr = np.empty((end_ndx-start_ndx, 2), dtype=np.str)
        save_arr = pd.DataFrame(save_arr, columns=['x', 'y'])
        for i in range(start_ndx, end_ndx):
            image_data =open(lines[i][1], 'rb')
            image = load_image(image_data)
            save_arr.values[i-start_ndx, 0] = image
            save_arr.values[i-start_ndx, 1] = int(lines[i][2].replace('\n',''))
            image_data.close()
        save_arr.to_csv(output_path, decimal=',', encoding='utf-8', index=False, index_label=False)
    fd.close()
TF_train_file_dir=des_path+'float_train/'
TF_eval_file_dir=des_path+'float_eval/'
mkdir(TF_train_file_dir)
mkdir(TF_eval_file_dir)
list_train_file_name=des_path+'list_train.txt'
list_eval_file_name=des_path+'list_val.txt'

convert_dataset(list_train_file_name, TF_train_file_dir)
convert_dataset(list_eval_file_name, TF_eval_file_dir)
