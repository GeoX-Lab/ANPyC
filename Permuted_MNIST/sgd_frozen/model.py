import tensorflow as tf
import numpy as np
import pickle
from math import ceil
from copy import deepcopy
# variable initialization functions
# def weight_variable(shape,name):
#     #initial = tf.truncated_normal(shape, stddev=0.1)
#     return tf.get_variable()

# def bias_variable(shape):
#     initial = tf.constant(0.1, shape=shape)
#     return tf.Variable(initial)

class Model:
    def __init__(self, x, y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,keep_prob):
        self.x = x  # input placeholder
        self.y1 = y1
        self.y2 = y2
        self.y3 = y3
        self.y4 = y4
        self.y5 = y5
        self.y6 = y6
        self.y7 = y7
        self.y8 = y8
        self.y9 = y9
        self.y10=y10
        self.keep_prob=keep_prob
        # simple 3-layer network 784-512-256-10
        with tf.variable_scope("weight"):
            W1 = tf.get_variable('w1',shape=[784, 512],initializer=tf.truncated_normal_initializer(stddev=0.1))
            b1 = tf.get_variable('b1',shape=[512],initializer=tf.constant_initializer(0.1))
            self.h1 = tf.nn.relu(tf.matmul(x, W1) + b1)  # hidden layer1

            W2 = tf.get_variable('w2',shape=[512,256],initializer=tf.truncated_normal_initializer(stddev=0.1))
            b2 = tf.get_variable('b2', shape=[256], initializer=tf.constant_initializer(0.1))
            self.h2 = tf.nn.relu(tf.matmul(self.h1, W2) + b2)  # hidden layer1
            self.dropout1 = tf.nn.dropout(self.h2, self.keep_prob)

        with tf.variable_scope("output_1"):
            W3 = tf.get_variable('w3', shape=[256,10], initializer=tf.truncated_normal_initializer(stddev=0.1))
            b3 = tf.get_variable('b3', shape=[10], initializer=tf.constant_initializer(0.1))
            self.out1 = tf.matmul(self.dropout1, W3) + b3  # output layer
            self.probs1 = tf.nn.softmax(self.out1)
        with tf.variable_scope("output_2"):
            W3 = tf.get_variable('w3', shape=[256, 10], initializer=tf.truncated_normal_initializer(stddev=0.1))
            b3 = tf.get_variable('b3', shape=[10], initializer=tf.constant_initializer(0.1))
            self.out2 = tf.matmul(self.dropout1, W3) + b3
            self.probs2 = tf.nn.softmax(self.out2)
        with tf.variable_scope("output_3"):
            W3 = tf.get_variable('w3', shape=[256, 10], initializer=tf.truncated_normal_initializer(stddev=0.1))
            b3 = tf.get_variable('b3', shape=[10], initializer=tf.constant_initializer(0.1))
            self.out3 = tf.matmul(self.dropout1, W3) + b3
            self.probs3 = tf.nn.softmax(self.out3)
        with tf.variable_scope("output_4"):
            W3 = tf.get_variable('w3', shape=[256, 10], initializer=tf.truncated_normal_initializer(stddev=0.1))
            b3 = tf.get_variable('b3', shape=[10], initializer=tf.constant_initializer(0.1))
            self.out4 = tf.matmul(self.dropout1, W3) + b3
            self.probs4 = tf.nn.softmax(self.out4)
        with tf.variable_scope("output_5"):
            W3 = tf.get_variable('w3', shape=[256, 10], initializer=tf.truncated_normal_initializer(stddev=0.1))
            b3 = tf.get_variable('b3', shape=[10], initializer=tf.constant_initializer(0.1))
            self.out5 = tf.matmul(self.dropout1, W3) + b3
            self.probs5 = tf.nn.softmax(self.out5)
        with tf.variable_scope("output_6"):
            W3 = tf.get_variable('w3', shape=[256, 10], initializer=tf.truncated_normal_initializer(stddev=0.1))
            b3 = tf.get_variable('b3', shape=[10], initializer=tf.constant_initializer(0.1))
            self.out6 = tf.matmul(self.dropout1, W3) + b3
            self.probs6 = tf.nn.softmax(self.out6)
        with tf.variable_scope("output_7"):
            W3 = tf.get_variable('w3', shape=[256, 10], initializer=tf.truncated_normal_initializer(stddev=0.1))
            b3 = tf.get_variable('b3', shape=[10], initializer=tf.constant_initializer(0.1))
            self.out7 = tf.matmul(self.dropout1, W3) + b3
            self.probs7 = tf.nn.softmax(self.out7)
        with tf.variable_scope("output_8"):
            W3 = tf.get_variable('w3', shape=[256, 10], initializer=tf.truncated_normal_initializer(stddev=0.1))
            b3 = tf.get_variable('b3', shape=[10], initializer=tf.constant_initializer(0.1))
            self.out8 = tf.matmul(self.dropout1, W3) + b3
            self.probs8 = tf.nn.softmax(self.out8)
        with tf.variable_scope("output_9"):
            W3 = tf.get_variable('w3', shape=[256, 10], initializer=tf.truncated_normal_initializer(stddev=0.1))
            b3 = tf.get_variable('b3', shape=[10], initializer=tf.constant_initializer(0.1))
            self.out9 = tf.matmul(self.dropout1, W3) + b3
            self.probs9 = tf.nn.softmax(self.out9)
        with tf.variable_scope("output_10"):
            W3 = tf.get_variable('w3', shape=[256, 10], initializer=tf.truncated_normal_initializer(stddev=0.1))
            b3 = tf.get_variable('b3', shape=[10], initializer=tf.constant_initializer(0.1))
            self.out10 = tf.matmul(self.dropout1, W3) + b3
            self.probs10 = tf.nn.softmax(self.out10)


            self.var_list = [W1, b1, W2, b2, W3, b3]
            #self.var_list=tf.trainable_variables()

        # vanilla single-task loss
        with tf.variable_scope("loss"):
            self.cross_entropy1 = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y1, logits=self.out1))
            self.cross_entropy2 = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y2, logits=self.out2))
            self.cross_entropy3 = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y3, logits=self.out3))
            self.cross_entropy4 = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y4, logits=self.out4))
            self.cross_entropy5 = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y5, logits=self.out5))
            self.cross_entropy6 = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y6, logits=self.out6))
            self.cross_entropy7 = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y7, logits=self.out7))
            self.cross_entropy8 = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y8, logits=self.out8))
            self.cross_entropy9 = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y9, logits=self.out9))
            self.cross_entropy10 = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y10, logits=self.out10))


        self.set_vanilla_loss1()
        self.set_vanilla_loss2()
        self.set_vanilla_loss3()
        self.set_vanilla_loss4()
        self.set_vanilla_loss5()
        self.set_vanilla_loss6()
        self.set_vanilla_loss7()
        self.set_vanilla_loss8()
        self.set_vanilla_loss9()
        self.set_vanilla_loss10()


        # performance metrics
        with tf.variable_scope("accuracy"):
            correct_prediction1 = tf.equal(tf.argmax(self.out1, 1), tf.argmax(self.y1, 1))
            correct_prediction2 = tf.equal(tf.argmax(self.out2, 1), tf.argmax(self.y2, 1))
            correct_prediction3 = tf.equal(tf.argmax(self.out3, 1), tf.argmax(self.y3, 1))
            correct_prediction4 = tf.equal(tf.argmax(self.out4, 1), tf.argmax(self.y4, 1))
            correct_prediction5 = tf.equal(tf.argmax(self.out5, 1), tf.argmax(self.y5, 1))
            correct_prediction6 = tf.equal(tf.argmax(self.out6, 1), tf.argmax(self.y6, 1))
            correct_prediction7 = tf.equal(tf.argmax(self.out7, 1), tf.argmax(self.y7, 1))
            correct_prediction8 = tf.equal(tf.argmax(self.out8, 1), tf.argmax(self.y8, 1))
            correct_prediction9 = tf.equal(tf.argmax(self.out9, 1), tf.argmax(self.y9, 1))
            correct_prediction10 = tf.equal(tf.argmax(self.out10, 1), tf.argmax(self.y10, 1))

            self.accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))
            self.accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))
            self.accuracy3 = tf.reduce_mean(tf.cast(correct_prediction3, tf.float32))
            self.accuracy4 = tf.reduce_mean(tf.cast(correct_prediction4, tf.float32))
            self.accuracy5 = tf.reduce_mean(tf.cast(correct_prediction5, tf.float32))
            self.accuracy6 = tf.reduce_mean(tf.cast(correct_prediction6, tf.float32))
            self.accuracy7 = tf.reduce_mean(tf.cast(correct_prediction7, tf.float32))
            self.accuracy8 = tf.reduce_mean(tf.cast(correct_prediction8, tf.float32))
            self.accuracy9 = tf.reduce_mean(tf.cast(correct_prediction9, tf.float32))
            self.accuracy10 = tf.reduce_mean(tf.cast(correct_prediction10, tf.float32))

    def star(self):
        # used for saving optimal weights after most recent task training
        self.star_vars = []

        for v in range(len(self.var_list)):
            self.star_vars.append(self.var_list[v].eval())


    def restore(self, sess):
        # reassign optimal weights for latest task
        if hasattr(self, "star_vars"):
            for v in range(len(self.var_list)):
                sess.run(self.var_list[v].assign(self.star_vars[v]))

    def set_vanilla_loss1(self):
        self.train_step1 = tf.train.GradientDescentOptimizer(0.001).minimize(self.cross_entropy1)
    def set_vanilla_loss2(self):
        # 选择待优化的参数
        output_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='output_2')
        #print(output_vars)
        self.train_step2 = tf.train.GradientDescentOptimizer(0.001).minimize(self.cross_entropy2,var_list=output_vars)
    def set_vanilla_loss3(self):
        output_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='output_3')
        self.train_step3 = tf.train.GradientDescentOptimizer(0.001).minimize(self.cross_entropy3,var_list=output_vars)
    def set_vanilla_loss4(self):
        output_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='output_4')
        self.train_step4 = tf.train.GradientDescentOptimizer(0.001).minimize(self.cross_entropy4,var_list=output_vars)
    def set_vanilla_loss5(self):
        output_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='output_5')
        self.train_step5 = tf.train.GradientDescentOptimizer(0.001).minimize(self.cross_entropy5,var_list=output_vars)
    def set_vanilla_loss6(self):
        output_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='output_6')
        self.train_step6 = tf.train.GradientDescentOptimizer(0.001).minimize(self.cross_entropy6,var_list=output_vars)
    def set_vanilla_loss7(self):
        output_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='output_7')
        self.train_step7 = tf.train.GradientDescentOptimizer(0.001).minimize(self.cross_entropy7,var_list=output_vars)
    def set_vanilla_loss8(self):
        output_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='output_8')
        self.train_step8 = tf.train.GradientDescentOptimizer(0.001).minimize(self.cross_entropy8,var_list=output_vars)
    def set_vanilla_loss9(self):
        output_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='output_9')
        self.train_step9 = tf.train.GradientDescentOptimizer(0.001).minimize(self.cross_entropy9,var_list=output_vars)
    def set_vanilla_loss10(self):
        output_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='output_10')
        self.train_step10 = tf.train.GradientDescentOptimizer(0.001).minimize(self.cross_entropy10,var_list=output_vars)