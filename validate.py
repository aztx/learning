# -*- coding: utf-8 -*-：
import tensorflow as tf
w1 = tf.Variable(tf.random_normal([2,3],stddev = 1),name = 'w1')
w2 = tf.Variable(tf.random_normal([2,2],stddev = 1),name = 'w2')
tf.assign(w1,w2,validate_shape=False)#词句用validate_shape=False来解除维度限制