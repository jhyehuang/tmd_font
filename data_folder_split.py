# -*- coding: utf-8 -*-
"""
从数据集中划分train和validation两个文件
train_test_split_ratio=0.1 or 0.2
Tree目录：
    data：
        train：
            folder1
            ......
            folder529
        validation:
            folder1
            ......
            folder529
"""
import os
import random
import PIL.Image as Image
import pandas as pd
import numpy as np


# 检查路径下面是否都是文件
def isfile(path):
    for folder in os.listdir(path):
        if not os.path.isdir(path+folder):
            os.remove(path+folder)


# 建立文件夹
def mkdir(path):
    """
    if folder is exists, or make new dir
    """
    isexists = os.path.exists(path)
    if not isexists:
        os.makedirs(path)
        print(path)
        print('success')
        return True
    else:
        print(path)
        print('folder is exist')
        return False

def load_image(image):
    img = Image.open(image)
    img = img.resize((128, 128))
    img = np.array(img)
    img = img / 255
    img = img.reshape(1,-1)  # reshape img to size(1, 128, 128, 1)
    return img

src_path = '../data/TMD/train/'
des_path = '../data/TMD/npz_train/'
def graph2npz():    
    label_list = []
    next_layer_files_list = []
    pathDir = os.listdir(src_path)
    label_file_name='label_list.txt'
    mkdir(des_path)
    mkdir(src_path)
    fd = open(des_path+label_file_name, 'w')
    for f in pathDir:
        label_list.append(f)
        next_layer_files_list=src_path+f
#        print( next_layer_files_list)
        graph_list=os.listdir(next_layer_files_list)
        for g in graph_list:
            image_name=next_layer_files_list+'/'+g
            fd.write('{},{},{}\n'.format(f, image_name, label_list.index(f)))
    fd.close()
    
def random_split_data():        
    test_train_split_ratio = 0.9
    _RANDOM_SEED = 0
    list_path = des_path+'label_list.txt'
    train_list_path = des_path+'list_train.txt'
    val_list_path = des_path+'list_val.txt'
    
    fd = open(list_path)
    lines = fd.readlines()
    fd.close()
    random.seed(_RANDOM_SEED)
    random.shuffle(lines)
    
    fd = open(train_list_path, 'w')
    for line in lines[:int(test_train_split_ratio * len(lines))]:
        fd.write(line)
    
    fd.close()
    fd = open(val_list_path, 'w')
    for line in lines[int(test_train_split_ratio * len(lines)):]:
        fd.write(line)
    
    fd.close()

if __name__ == '__main__':
    graph2npz()
    random_split_data()


