
# coding: utf-8

# In[1]:


"""A very simple MNIST classifier.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
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


def records_to(file, num_threads=2, num_epochs=2, batch_size=2, min_after_dequeue=2):
    reader = tf.TFRecordReader()
    file_queue = tf.train.string_input_producer(file, num_epochs=num_epochs,)
    _, example = reader.read(file_queue)
    features_dict = tf.parse_single_example(example, 
        features={
            'image/class/label': tf.FixedLenFeature([], tf.int),
            'image/encoded': tf.FixedLenFeature([], tf.float32),
            'image/format': tf.FixedLenFeature([], tf.string),
            'image/height': tf.FixedLenFeature([], tf.string),
            'image/width': tf.FixedLenFeature([], tf.string)
        })
    # n: Tensor("ParseSingleExample/Squeeze_name:0", shape=(), dtype=string)
    label = features_dict['image/class/label']
    image = features_dict['image/encoded']
    image_format = features_dict['image/format']
    height = features_dict['image/height']
    width = features_dict['image/width']
    label, image, image_format,height,width = tf.train.shuffle_batch(
        [label, image, image_format,height,width],
        batch_size=batch_size,
        num_threads=num_threads,
        capacity = min_after_dequeue + 3 * batch_size,
        min_after_dequeue = min_after_dequeue
    )
    # 数据格式为Tensor
    return label, image, image_format,height,width

def train():
    label, image, image_format,height,width = records_to(data_file)
    with tf.Session() as sess:
        tf.group(tf.global_variables_initializer(),
            tf.local_variables_initializer()).run()
        tf.train.start_queue_runners(sess=sess)
        label_val, image_val, image_format_val , height_val, width_val \
        = sess.run([label, image, image_format,height,width])
        print(label_val, image_val, image_format_val , height_val, width_val)

record_iterator=tf.python_io.tf_record_iterator(path=data_file)
for string_record in record_iterator:
    
    example = tf.train.Example()
    example.ParseFromString(string_record)
   # print (example)
    n = example.features.feature['image/class/label'].int64_list.value
    t = example.features.feature['image/encoded'].bytes_list.value
    d = example.features.feature['image/format'].bytes_list.value
    e = example.features.feature['image/height'].int64_list .value
    f = example.features.feature['image/width'].int64_list .value
    
    print(n,  d,e,f)
    break



x = tf.placeholder(tf.float32, [None, 128*128])
y_ = tf.placeholder(tf.float32, [None, 100])



# ## 权重初始化

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# # 卷积和池化

# In[5]:


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')


# # 第一层卷积

# In[7]:


W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1,128,128,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


# # 第二层卷积

# In[8]:


W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


# # 密集连接层
# 999  98.66
# 784  98.11
# 

# In[42]:


W_fc1 = weight_variable([32 * 32 * 64, 999])
b_fc1 = bias_variable([999])
h_pool2_flat = tf.reshape(h_pool2, [-1, 32*32*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


# #  Dropout

# In[43]:


keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# # 输出层

# In[44]:


W_fc2 = weight_variable([999, 100])
b_fc2 = bias_variable([100])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# # 训练和评估模型

# ## 计算交叉熵

# In[49]:


cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4*0.9).minimize(cross_entropy) 
#train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy) 
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# In[50]:


with tf.Session() as sess:
    init=tf.global_variables_initializer()
    sess.run(init)
    for i in range(25000):
        batch = mnist.train.next_batch(50)
        if i%100==0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    print ("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


