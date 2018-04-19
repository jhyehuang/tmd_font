# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 16:42:40 2018

@author: admin
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from data_folder_split import load_image 
import numpy as np
import pandas as pd


FLAGS = None

data_file = '../data/TMD/npz_train/train/data_00000-of-00005.tfrecord'


def records_to(file, num_threads=2, num_epochs=2, batch_size=30, min_after_dequeue=1000):
    file_queue = tf.train.string_input_producer([file],num_epochs=num_epochs,)
    reader = tf.TFRecordReader()
    _, example = reader.read(file_queue)
    features_dict = tf.parse_single_example(example, 
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string),
            'image_format': tf.FixedLenFeature([], tf.string),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64)
        })
    label =  tf.cast(features_dict['label'],tf.int64)
    image = tf.decode_raw(features_dict['image'], tf.uint8)
    
    #image = tf.cast(image, tf.float32) - 0.5
    image_format =  tf.cast(features_dict['image_format'],tf.string)
    height =  tf.cast(features_dict['height'],tf.int64)
    width =  tf.cast(features_dict['width'],tf.int64)
    image=image.set_shape([None,])
    print (label,image,height,width)
    label_batch, image_batch,height_batch,width_batch = tf.train.shuffle_batch(
        [label, image,height,width],
        batch_size=30,
        num_threads=2,
        capacity =2000,
        min_after_dequeue = 1000)
    # 数据格式为Tensor
    return label_batch, image_batch,height_batch,width_batch

def train():
    label, image,height,width = records_to(data_file)
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
#        tf.group(tf.global_variables_initializer(),
#            tf.local_variables_initializer()).run()
        tf.train.start_queue_runners(sess=sess)
        label_val, image_val , height_val, width_val \
        = sess.run([label, image,height,width])
        print(label_val, image_val , height_val, width_val)
        
train()