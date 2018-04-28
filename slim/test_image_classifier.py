# -*- coding: utf-8 -*-

# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import numpy as np

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory
from tensorflow.python.platform import tf_logging as logging
import time
from nets.densenet import densenet_arg_scope
import pandas as pd
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step


slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'batch_size', 1, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'test_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 8,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

tf.app.flags.DEFINE_string(
    'labels_file', None, 'labels_file')

FLAGS = tf.app.flags.FLAGS

#State the number of epochs to evaluate
num_epochs = 1

#State the labels file and read it

labels = open(FLAGS.labels_file, 'rb')

#Create a dictionary to refer each label to their string name
labels_to_name = {}
for line in labels:
    label, string_name = line.split(b':')
    string_name = string_name[:-1] #Remove newline
    string_name=str(string_name, encoding='utf-8')
    labels_to_name[int(label)] = string_name


def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')
        
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():

        ######################
        # Select the dataset #
        ######################
        dataset = dataset_factory.get_dataset(FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)
    
        ####################
        # Select the model #
        ####################
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=(dataset.num_classes - FLAGS.labels_offset),
            is_training=False)
    
        ##############################################################
        # Create a dataset provider that loads data from the dataset #
        ##############################################################
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            shuffle=False,
            common_queue_capacity=2 * FLAGS.batch_size,
            common_queue_min=FLAGS.batch_size)
        [image, file_name] = provider.get(['image', 'file_name'])
    
        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=False)
    
        eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size
    
        image = image_preprocessing_fn(image, eval_image_size, eval_image_size)
    
        images, file_names = tf.train.batch(
            [image, file_name],
            batch_size=FLAGS.batch_size,
            num_threads=FLAGS.num_preprocessing_threads,
            capacity=5 * FLAGS.batch_size)
        
        ####################
        # Define the model #
        ####################
    #    logits, end_points = network_fn(images)
    
    #    predictions = tf.argmax(logits, 1)
    
        # TODO(sguada) use num_epochs=1
        num_batches_per_epoch = math.ceil(dataset.num_samples / float(FLAGS.batch_size))
        num_steps_per_epoch = num_batches_per_epoch

        with slim.arg_scope(densenet_arg_scope()):
            logits, end_points = network_fn(images)    
            
        print (FLAGS.checkpoint_path)
        if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        else:
            checkpoint_path = FLAGS.checkpoint_path
        tf.logging.info('Evaluating %s' % checkpoint_path)

    
        # #get all the variables to restore from the checkpoint file and create the saver function to restore
        variables_to_restore = slim.get_variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        def restore_fn(sess):
            return saver.restore(sess, checkpoint_path)

        #Just define the metrics to track without the loss or whatsoever
        #predictions_e = tf.argmax(logits, 100)
        predictions_e = end_points['predictions']

        #top_k_pred = tf.nn.top_k(predictions_e, k=5)
        top_k_pred = tf.nn.top_k(predictions_e, k=1)
#        top_k_pred = tf.nn.top_k(logits, k=5)
        tf.logging.info(' logits=%s' % logits)
        tf.logging.info(' end_points=%s' % end_points)
        #Create the global step and an increment op for monitoring
        global_step = get_or_create_global_step()
        global_step_op = tf.assign(global_step, global_step + 1) #no apply_gradient method so manually increasing the global_step
        
        file_names_all = []
        predictions_all = []
        all_key={}
        c_key={}
        #Create a evaluation step function
        def eval_step(sess, top_k_pred,file_names,global_step):
            '''
            Simply takes in a session, runs the metrics op and some logging information.
            '''
            start_time = time.time()
#            global_step_count, predictions_ = sess.run([global_step_op, predictions])
#            p_file_name=sess.run(file_names)
            global_step_count,p_file_name,output=sess.run([global_step_op,file_names,top_k_pred])
#            output = sess.run(top_k_pred)  
            probability = np.array(output[0]).flatten()  # 取出概率值，将其展成一维数组  
            index = np.array(output[1]).flatten()
            tf.logging.info(' %s' % probability)
            tf.logging.info(' %s' % index)
            predictions_=probability
            time_elapsed = time.time() - start_time
            #Log some information
            logging.info('Global Step %s: Streaming Accuracy: (%.2f sec/step)', global_step_count, time_elapsed)

            return  predictions_,index,p_file_name


        #Get your supervisor
        sv = tf.train.Supervisor(logdir = FLAGS.test_dir, summary_op = None, saver = None, init_fn = restore_fn)

        #Now we are ready to run in one session
        #config = tf.ConfigProto(device_count={'GPU':0}) # mask GPUs visible to the session so it falls back on CPU
        with sv.managed_session() as sess:
            for step in range(num_steps_per_epoch * num_epochs):
                sess.run(sv.global_step)
                #print vital information every start of the epoch as always
                predictions_ ,font_index,file_names_= eval_step(sess,top_k_pred,file_names, global_step = sv.global_step)
                my_predictions=[]
                for x in font_index:
                    my_predictions.append(labels_to_name[int(x)])
                logging.info(file_names_)
                logging.info(str(file_names_[0],'utf-8'))
                my_file_name=str(file_names_[0],'utf-8')
                logging.info(my_predictions)
                logging.info('my_file_name={}'.format(my_file_name))
                if my_file_name not in all_key:
                    all_key[my_file_name]=predictions_all
                else:
                    c_key[my_file_name]=predictions_all
                file_names_all = np.append(file_names_all, my_file_name)
                predictions_all = np.append(predictions_all, ''.join(my_predictions))

            #At the end of all the evaluation, show the final accuracy
            logging.info('总处理不重复的文件数:'+str(len(all_key)))
            logging.info('总处理重复的文件数:'+str(len(c_key)))
            logging.info('Model evaluation has completed! Visit TensorBoard for more information regarding your evaluation.')
        rpt = pd.DataFrame({'filename':file_names_all,'label':predictions_all})  
        rpt.to_csv(FLAGS.test_dir+'/chinese_font.csv',encoding = "utf-8",index=False)
        

if __name__ == '__main__':
  tf.app.run()
