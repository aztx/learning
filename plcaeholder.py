# -*- coding: utf-8 -*-：
import tensorflow as tf

w1 = tf.Variable(tf.random_normal([2,3],stddev = 1))
w2 = tf.Variable(tf.random_normal([3,1],stddev = 1))

#定义placeholder作为存放输入数据的地方。这里维度也不一定要定义。
#但如果维度是确定的，那么给出维度可以降低出错率
#x = tf.placeholder(tf.float32, shape=(1,2),name="input")
x = tf.placeholder(tf.float32, shape=(3,2),name="input")
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)

#用到tf.placeholder()时,需要feed_dict来指定x的取值。
# feed_dicts是一个字典，在字典中需要给出每一个用到的placeholder的取值
#print(sess.run(y,feed_dict = {x: [[0.7,0.9]]}))#为何y和feed_dict得一起run

print(sess.run(y,feed_dict = {x: [[0.7,0.9], [0.1,0.4], [0.5,0.8]]}))#为何y和feed_dict得一起run
