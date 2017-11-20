# -*- coding: utf-8 -*-：
import tensorflow as tf

matrix1 = tf.constant([[3.,3.]])
matrix2 = tf.constant([[2.],[2.]])

product = tf.matmul(matrix1,matrix2)

#启动默认图
sess = tf.Session()
result = sess.run(product)
print result

sess.close()

#with tf.Session() as sess:
 #   with tf.device("/cpu:0"):
  #      matrix1 = tf.constant([[3.,3.]])
   #     matrix2 = tf.constant([[2.],[2.]])
    #    product = tf.matmul(matrix1,matrix2)
