#coding=utf-8
'''
A linear regression learning algorithm example using TensorFlow library.
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
写点中文试试
'''

from __future__ import print_function

import tensorflow as tf
import numpy
import matplotlib.pyplot as plt

# 生成简单的随机数据，包括随机值，随机正态样本，返回随机整数或浮点数，返回随机排列，返回随机的各种分布，随机数生成器
rng = numpy.random

# Parameters
learning_rate = 0.01  # 梯度下降的步长
training_epochs = 1000   #训练周期
display_step = 50

# Training Data
train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_X.shape[0]

# tf Graph Input  图像输入节点和输出节点
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights  生成随机权重和偏置
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

# Construct a linear model 在TensoorFlow中，所有的操作op，变量都视为节点，tf.add() 的意思就是在tf的默认图中添加一个op，这个op是用来做加法操作的。
# 构造一个op节点
pred = tf.add(tf.mul(X, W), b)

# Mean squared error
# 计算L2 loss，其中tf.reduce_sum(t,reduction_indices)为规约函数，规约向量的sum值作为返回值，reduction_indices为0时按列规约，为1时按行规约
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
# Gradient descent   使用步长为0.01的梯度下降法训练模型,这个运算的目的就是最小化loss。
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables  用于初始化变量。此操作不会立即执行。需要通过sess来将数据流动起来
init = tf.initialize_all_variables()

# Launch the graph  所有的运算都应在在session中进行
with tf.Session() as sess:
    #自动开启一个session
    sess.run(init)

    # Fit all training data  训练周期是1000
    for epoch in range(training_epochs):
        #zip()是Python的一个内建函数，它接受一系列可迭代的对象作为参数，将对象中对应的元素打包成一个个tuple（元组），然后返回由这些tuples组成的list（列表）
        for (x, y) in zip(train_X, train_Y):
            #运行会话，输入数据，并计算节点,optimizer是输出，feed_dict是输入
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # Display logs per epoch step  每50步一个展示
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    # Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()

    # Testing example, as requested (Issue #2)
    test_X = numpy.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
    test_Y = numpy.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

    print("Testing... (Mean square loss Comparison)")
    testing_cost = sess.run(
        tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
        feed_dict={X: test_X, Y: test_Y})  # same function as cost above
    print("Testing cost=", testing_cost)
    print("Absolute mean square loss difference:", abs(
        training_cost - testing_cost))

    plt.plot(test_X, test_Y, 'bo', label='Testing data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()