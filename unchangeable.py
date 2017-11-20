# -*- coding: utf-8 -*-：
import tensorflow as tf
w1 = tf.Variable(tf.random_normal([2,3],stddev = 1),name ='w1')
w2 = tf.Variable(tf.random_normal([2,3],dtype = tf.float64, stddev=1),name='w2')
w1.assign(w2)#将w2中的值赋给w1,此句报错