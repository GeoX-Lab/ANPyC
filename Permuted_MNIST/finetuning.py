import tensorflow as tf
import numpy as np
from copy import deepcopy
from tensorflow.examples.tutorials.mnist import input_data
import random
import matplotlib.pyplot as plt



def permute_mnist(mnist,per_task):
    perm_inds = list(range(mnist.train.images.shape[1]))
    random.seed(per_task)
    random.shuffle(perm_inds)
    mnist2 = deepcopy(mnist)
    sets = ["train", "validation", "test"]
    for set_name in sets:
        this_set = getattr(mnist2, set_name)  # shallow copy
        this_set._images = np.transpose(np.array([this_set.images[:, c] for c in perm_inds]))
    return mnist2

def weight_variable(shape):
	with tf.name_scope('weights'):
		initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	with tf.name_scope('biases'):
		initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

class Model:
	def __init__(self, x):
		# create network 784-50-50-10
		in_dim = int(x.get_shape()[1]) # 784 for MNIST

		self.x = x # input placeholder

		keep_prob = 0.5
		# simple 3-layer network
		self.W1 = weight_variable([in_dim,512])
		self.b1 = bias_variable([512])
		self.W2 = weight_variable([512, 256])
		self.b2 = bias_variable([256])

		self.h1 = tf.nn.relu(tf.matmul(self.x, self.W1) + self.b1)  # hidden layer
		self.h1_drop = tf.nn.dropout(self.h1, keep_prob)

		self.h2 = tf.nn.relu(tf.matmul(self.h1_drop,self.W2) + self.b2) # hidden layer
		self.h2_drop = tf.nn.dropout(self.h2,keep_prob)
		self.params = [self.W1,self.b1,self.W2,self.b2]
		return

def optimizer(lr_start,cost,varlist,steps,epoches):
	learning_rate = tf.train.exponential_decay(lr_start, global_step=steps, decay_steps=epoches, decay_rate=0.9,staircase=True) # lr decay=1
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost,global_step=steps, var_list=varlist)
	return train_step,learning_rate

def clone_net(net_old, net):
	for i in range(len(net.params)):
		assign_op = net_old.params[i].assign(net.params[i])
		sess.run(assign_op)


mnist = input_data.read_data_sets("/tmp/", one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])
model = Model(x)
model_old = Model(x)

# train and test
num_tasks_to_run = 10
num_epochs_per_task = 50
minibatch_size = 256
epoches = 10*mnist.train.num_examples/minibatch_size #decay per 10 epoch
learning_r = 0.1

# Generate the tasks specifications as a list of random permutations of the input pixels.
task_permutation = []
for task in range(num_tasks_to_run):
	mnist_ = permute_mnist(mnist,task)
	task_permutation.append(mnist_)

y_ = tf.placeholder(tf.float32, shape=[None, 10])
out_dim = int(y_.get_shape()[1])  # 10 for MNIST
lr_steps = tf.Variable(0)

# expand output layer
W_output = []
b_output = []
task_output = []
for task in range(num_tasks_to_run):
	W = weight_variable([256, out_dim])
	b = bias_variable([out_dim])
	W_output.append(W)
	b_output.append(b)
for task in range(num_tasks_to_run):
	output = tf.matmul(model.h2_drop, W_output[task]) + b_output[task]
	task_output.append(output)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)
sess.run(tf.global_variables_initializer())

per_avg_performance = []
first_performance = []
last_performance = []
tasks_acc = []

for task in range(num_tasks_to_run):
    # print "Training task: ",task+1,"/",num_tasks_to_run
    print("\t task ", task + 1)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=task_output[task]))
    train_op,learning_rate = optimizer(lr_start=learning_r, cost=cost, varlist=[model.params, W_output[task], b_output[task]],steps=lr_steps,epoches=epoches)

    sess.run(tf.variables_initializer([lr_steps]))

    for epoch in range(num_epochs_per_task):
        if epoch % 5 == 0:
            print("\t Epoch ", epoch)

        for i in range(int(mnist.train.num_examples / minibatch_size) + 1):
            batch = task_permutation[task].train.next_batch(minibatch_size)
            sess.run(train_op, feed_dict={x: batch[0], y_: batch[1]})
    # Print test set accuracy to each task encountered so far
    avg_accuracy = 0.0
    for test_task in range(task + 1):
        correct_prediction = tf.equal(tf.argmax(task_output[test_task], 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        test_data = task_permutation[test_task].test

        acc = sess.run(accuracy, feed_dict={x: test_data.images, y_: test_data.labels}) * 100.0
        avg_accuracy += acc

        if test_task == 0:
            first_performance.append(acc)
        if test_task == task:
            last_performance.append(acc)
        tasks_acc.append(acc)
        print("Task: ", test_task, " \tAccuracy: ", acc)

    avg_accuracy = avg_accuracy / (task + 1)
    print("Avg Perf: ", avg_accuracy)



# save best model-test log
best_acc = tasks_acc
file= open('log-fintuning.txt', 'w')
for fp in best_acc:
	file.write(str(fp))
	file.write('\n')

file.write(str(per_avg_performance))
file.close()

sess.close()