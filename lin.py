# -*- coding: utf-8 -*-：
import tensorflow as tf
import numpy as np

#creat data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1+0.3

#create tensorflow structure start#

Weights=tf.Variable(tf.random_uniform([1],-1.0,1.0))#bianliang
biases=tf.Variable(tf.zeros([1]))


y=Weights*x_data + biases

#tensorflow 内部函数
loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()#初始化变量，激活前面定义动变量

##creat tensorflow structure start#

sess = tf.Session()    #建立会话
sess.run(init)         #激活init


for step in range(201):
    sess.run(train)
    if step %20 ==0:
        print(step,sess.run(Weights),sess.run(biases))