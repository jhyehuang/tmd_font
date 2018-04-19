# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 09:53:40 2018

@author: huang
"""
import sys
sys.path.insert(0, '../models/research/slim/')
from datasets import dataset_utils
import math
import os
import tensorflow as tf
from  data_folder_split import mkdir ,des_path,load_image
#import adjust_pic as ap

def convert_dataset(list_path, output_dir, _NUM_SHARDS=5):
    fd = open(list_path)
    lines=[]
    for line in fd:
        line=line.split(',')
        lines.append(line)
    fd.close()
    num_per_shard = int(math.ceil(len(lines) / float(_NUM_SHARDS)))
    with tf.Graph().as_default():
        with tf.Session('') as sess:
            for shard_id in range(_NUM_SHARDS):
                output_path = os.path.join(output_dir,
                    'data_{:05}-of-{:05}.tfrecord'.format(shard_id, _NUM_SHARDS))
                print(output_path)
                tfrecord_writer = tf.python_io.TFRecordWriter(output_path)
                start_ndx = shard_id * num_per_shard
                end_ndx = min((shard_id + 1) * num_per_shard, len(lines))
                for i in range(start_ndx, end_ndx):
                    image_data = tf.gfile.FastGFile(lines[i][1], 'rb').read()
                    image =tf.image.decode_jpeg(image_data) 
                    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
#                    image=tf.image.resize_images(image, [100, 100], method=0)
                    image = sess.run(image)
                    image_raw = image.tostring() 
                    height, width = image.shape[0], image.shape[1]
                    print('{}/{}'.format(i,end_ndx))
                    feature={
                        'image':tf.train.Feature(bytes_list = tf.train.BytesList(value=[image_raw])),
                        'image_format':tf.train.Feature(bytes_list = tf.train.BytesList(value=[b'jpg'])), 
                        'height':tf.train.Feature(int64_list = tf.train.Int64List(value=[height])), 
                        'width':tf.train.Feature(int64_list = tf.train.Int64List(value=[width])),
                        'label':tf.train.Feature(int64_list = tf.train.Int64List(value=[int(lines[i][2].replace('\n',''))]))}
                    features = tf.train.Features(feature=feature)
                    example = tf.train.Example(features=features)
                    tfrecord_writer.write(example.SerializeToString())
                tfrecord_writer.close()

TF_train_file_dir=des_path+'train/'
TF_eval_file_dir=des_path+'eval/'
mkdir(TF_train_file_dir)
mkdir(TF_eval_file_dir)
list_train_file_name=des_path+'list_train.txt'
list_eval_file_name=des_path+'list_val.txt'

convert_dataset(list_train_file_name, TF_train_file_dir)
convert_dataset(list_eval_file_name, TF_eval_file_dir)
