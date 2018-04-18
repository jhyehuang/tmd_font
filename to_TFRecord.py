# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 09:53:40 2018

@author: huang
"""
import sys
sys.path.insert(0, '../models/slim/')
from datasets import dataset_utils
import math
import os
import tensorflow as tf
from  data_folder_split import mkdir ,des_path

def convert_dataset(list_path, data_dir, output_dir, _NUM_SHARDS=5):
    fd = open(list_path)
    lines = [line.split() for line in fd]
    fd.close()
    print(lines)
#    num_per_shard = int(math.ceil(len(lines) / float(_NUM_SHARDS)))
#    with tf.Graph().as_default():
#        decode_jpeg_data = tf.placeholder(dtype=tf.string)
#        decode_jpeg = tf.image.decode_jpeg(decode_jpeg_data, channels=3)
#        with tf.Session('') as sess:
#            for shard_id in range(_NUM_SHARDS):
#                output_path = os.path.join(output_dir,
#                    'data_{:05}-of-{:05}.tfrecord'.format(shard_id, _NUM_SHARDS))
#                tfrecord_writer = tf.python_io.TFRecordWriter(output_path)
#                start_ndx = shard_id * num_per_shard
#                end_ndx = min((shard_id + 1) * num_per_shard, len(lines))
#                for i in range(start_ndx, end_ndx):
#                    sys.stdout.write('\r>> Converting image {}/{} shard {}'.format(
#                        i + 1, len(lines), shard_id))
#                    sys.stdout.flush()
#                    image_data = tf.gfile.FastGFile(os.path.join(data_dir, lines[i][0]), 'rb').read()
#                    image = sess.run(decode_jpeg, feed_dict={decode_jpeg_data: image_data})
#                    height, width = image.shape[0], image.shape[1]
#                    example = dataset_utils.image_to_tfexample(
#                        image_data, b'jpg', height, width, int(lines[i][1]))
#                    tfrecord_writer.write(example.SerializeToString())
#                tfrecord_writer.close()
#    sys.stdout.write('\n')
#    sys.stdout.flush()

TF_train_file_dir=des_path+'train/'
TF_eval_file_dir=des_path+'eval/'
mkdir(des_path+'train')
mkdir(des_path+'val')
list_train_file_name=des_path+'list_train.txt'
list_eval_file_name=des_path+'list_val.txt'

convert_dataset(list_train_file_name, 'flower_photos', TF_train_file_dir)
convert_dataset(list_eval_file_name, 'flower_photos', TF_train_file_dir)
