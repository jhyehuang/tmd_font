
import tensorflow as tf
saver = tf.train.import_meta_graph('/home/zhijie.huang/github/data/TMD/train_set/graph.pbtxt')
sess = tf.Session()
saver.resore(sess, '/home/zhijie.huang/github/data/TMD/train_set/model.ckpt-18753')
graph = sess.graph
print([node.name for node in graph.as_graph_def().node])