# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 12:53:27 2018

@author: admin
"""
import tensorflow as tf
from data_folder_split import load_image 
import numpy as np
import pandas as pd

record_iterator=tf.python_io.tf_record_iterator(path=data_file)
for string_record in record_iterator:
    
    example = tf.train.Example()
    example.ParseFromString(string_record)
   # print (example)
    print (example)
    n = example.features.feature['label'].int64_list.value
    t = example.features.feature['image'].bytes_list.value
    d = example.features.feature['image_format'].bytes_list.value
    e = example.features.feature['height'].int64_list .value
    f = example.features.feature['width'].int64_list .value
    
    print(n,  d,e,f)
    break