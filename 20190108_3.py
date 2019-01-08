# -*- coding: utf-8 -*-
"""
用CNN预测圆周上的点
"""

import tensorflow as tf

import numpy as np

train_total = 1000
test_total = 100

batch_size = 100
batch_num = train_total // batch_size


def build_data(n,m):
    x = np.random.rand(n)*50
    y = np.random.rand(m)*50
    x_train = np.zeros([n,10,2]).astype('float32')
    x_test = np.zeros([m,10,2]).astype('float32')
    y_train = np.zeros([n,2]).astype('float32')
    y_test = np.zeros([m,2]).astype('float32')
    for i in range(n):
        for j in range(10):
            x_train[i,j,:] = np.array([np.sin(x[i]+j),np.cos(x[i]+j)])
        y_train[i] = np.array([np.sin(x[i]+10),np.cos(x[i]+10)])
        
    for i in range(m):
        for j in range(10):
            x_test[i,j,:] = np.array([np.sin(y[i]+j),np.cos(y[i]+j)])
        y_test[i] = np.array([np.sin(y[i]+10),np.cos(y[i]+10)])

    return x_train,y_train,x_test,y_test


x_train,y_train,x_test,y_test = build_data(train_total,test_total)


tf.reset_default_graph()

x = tf.placeholder(tf.float32,shape=[None,10,2])

y = tf.placeholder(tf.float32,shape=[None,2])


z = tf.reshape(x,[-1,10,2,1])

filter1 = tf.Variable(tf.ones(shape=[2,2,1,10]))

layer1 = tf.nn.conv2d(z,filter1,strides=[1,1,1,1],padding='SAME') 

layer2 = tf.reshape(layer1,[-1,200])

layer3 = tf.contrib.layers.fully_connected(layer2,12)



pred = tf.contrib.layers.fully_connected(layer3,2,tf.nn.tanh)

loss = tf.reduce_mean(tf.square(pred-y))

optimizer = tf.train.AdamOptimizer(1e-2).minimize(loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(500):
        for j in range(batch_num):
            xx = x_train[j*batch_size:(j+1)*batch_size]
            yy = y_train[j*batch_size:(j+1)*batch_size]
            sess.run(optimizer,feed_dict={x:xx, y:yy})
        print("epoch =",epoch,"loss =",sess.run(loss,feed_dict={x:x_test,y:y_test}))
        
    print(sess.run(pred,feed_dict={x:x_test[9:10]}))
    print(y_test[9:10])


    
