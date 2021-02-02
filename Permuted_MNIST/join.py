import tensorflow as tf
import numpy as np
from copy import deepcopy
from tensorflow.examples.tutorials.mnist import input_data
import random


def permute_mnist(mnist, per_task):
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
		in_dim = int(x.get_shape()[1])  # 784 for MNIST
		N1 = 512
		N2 = 256
		self.x = x  # input placeholder

		keep_prob = 0.5
		# simple 3-layer network
		self.W1 = weight_variable([in_dim, N1])
		self.b1 = bias_variable([N1])
		self.W2 = weight_variable([N1, N2])
		self.b2 = bias_variable([N2])

		self.h1 = tf.nn.relu(tf.matmul(self.x, self.W1) + self.b1)  # hidden layer
		self.h1_drop = tf.nn.dropout(self.h1, keep_prob)

		self.h2 = tf.nn.relu(tf.matmul(self.h1_drop, self.W2) + self.b2)  # hidden layer
		self.h2_drop = tf.nn.dropout(self.h2, keep_prob)
		self.params = [self.W1, self.b1, self.W2, self.b2]
		return


def optimizer(lr_start, cost, varlist, steps, epoches):
	learning_rate = tf.train.exponential_decay(lr_start, global_step=steps, decay_steps=epoches, decay_rate=0.9,staircase=True)  # lr decay=1
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, global_step=steps, var_list=varlist)
	return train_step, learning_rate


def shuffle_data(X, y):
	s = np.arange(len(X))
	np.random.shuffle(s)
	X = X[s]
	y = y[s]
	return X, y


def label_to_one_hot(y, C):
	return np.eye(C)[y.reshape(-1)]


def yield_mb(X, y, batchsize=256, shuffle=False, one_hot=False):
	assert len(X) == len(y)
	if shuffle:
		X, y = shuffle_data(X, y)
	if one_hot:
		y = label_to_one_hot(y, 2)
	# Only complete batches are submitted
	for i in range(len(X) // batchsize):
		yield X[i * batchsize:(i + 1) * batchsize], y[i * batchsize:(i + 1) * batchsize]


def one_hot_expand(label_list,T):
	for i in range(len(label_list)):
		shape = label_list[i].shape[0]
		if i ==0:
			label_list[i] = np.hstack((label_list[i],np.zeros([shape,T * (T-i-1)])))
		else:
			label_list[i] = np.hstack((label_list[i],np.zeros([shape,T * (T-i-1)])))
			label_list[i] = np.hstack((np.zeros([shape,T * i]),label_list[i]))
	return label_list

def list_array(list):
	a = list[0]
	for i in range(1, len(list)):
		a = np.vstack((a,list[i]))
	return a





mnist = input_data.read_data_sets("/tmp/", one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])
model = Model(x)

# train and test
num_tasks_to_run = 10
num_epochs_per_task = 50
minibatch_size = 256
learning_r = 0.1
epoches = 10 * mnist.train.num_examples*10 / minibatch_size  # decay per 10 epoch

# Generate the tasks specifications as a list of random permutations of the input pixels.
mnist = input_data.read_data_sets("/tmp/", one_hot=True)

task_permutation = []
for task in range(num_tasks_to_run):
	# task_permutation.append( np.random.permutation(784) )
	mnist_ = permute_mnist(mnist, task)
	task_permutation.append(mnist_)

label_list_train = []
img_list_train = []
label_list_test = []
img_list_test = []
for j in range(len(task_permutation)):
	label_list_train.append(task_permutation[j].train.labels)
	img_list_train.append(task_permutation[j].train.images)
	label_list_test.append(task_permutation[j].test.labels)
	img_list_test.append(task_permutation[j].test.images)

label_list_train = list_array(one_hot_expand(label_list_train,T=10))
label_list_test = one_hot_expand(label_list_test,T=10)
img_list_train = list_array(img_list_train)
img_list_test = img_list_test

y_ = tf.placeholder(tf.float32, shape=[None, 100])
out_dim = int(y_.get_shape()[1])  # 10 for MNIST
lr_steps = tf.Variable(0)

# expand output layer
W_output = weight_variable([256, out_dim])
b_output = bias_variable([out_dim])
task_output = tf.matmul(model.h2_drop, W_output) + b_output

# loss function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=task_output))
train_op, learning_rate = optimizer(lr_start=learning_r, cost=cost, varlist=[model.params, W_output, b_output],steps=lr_steps, epoches=epoches)

correct_prediction = tf.equal(tf.argmax(task_output, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
## Initialize session
config = tf.ConfigProto()

config.gpu_options.allow_growth = True

sess = tf.InteractiveSession(config=config)
sess.run(tf.global_variables_initializer())

for epoch in range(num_epochs_per_task):
	if epoch % 5 == 0:
		print("\t Epoch ", epoch)

	for images, labels in yield_mb(img_list_train, label_list_train, batchsize=256, shuffle=True, one_hot=False):
		sess.run(train_op, feed_dict={x: images, y_: labels})

tasks_acc = []
avg_accuracy = 0.0
for test_task in range(num_tasks_to_run):
	acc = sess.run(accuracy, feed_dict={x: img_list_test[test_task], y_: label_list_test[test_task]}) * 100.0
	avg_accuracy += acc


	tasks_acc.append(acc)
	print("Task: ", test_task, " \tAccuracy: ", acc)

avg_accuracy = avg_accuracy / (task + 1)
print("Avg Perf: ", avg_accuracy)


# save best model-test log
best_acc = tasks_acc
file = open('log-join.txt', 'w')
for fp in best_acc:
	file.write(str(fp))
	file.write('\n')
file.write(str(avg_accuracy))
file.close()

sess.close()