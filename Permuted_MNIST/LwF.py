import tensorflow as tf
import numpy as np
from copy import deepcopy
from tensorflow.examples.tutorials.mnist import input_data
import random
import matplotlib.pyplot as plt


# load data
mnist = input_data.read_data_sets("/tmp/", one_hot=True)

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

# network
# variable initialization functions
def weight_variable(shape):
	with tf.name_scope('weights'):
		initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	with tf.name_scope('biases'):
		initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

class Model:
	def __init__(self, x,keep_prob=0.5):
		# create network 784-50-50-10
		in_dim = int(x.get_shape()[1]) # 784 for MNIST

		self.x = x # input placeholder

		# simple 3-layer network
		self.W1 = weight_variable([in_dim,512])
		self.b1 = bias_variable([512])
		self.W2 = weight_variable([512, 256])
		self.b2 = bias_variable([256])
		# W3 = weight_variable([256,out_dim])
		# b3 = bias_variable([out_dim])

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

def lwf_criterion(t,targets_old,outputs,targets,lamb=10,T=2):
	# Knowledge distillation loss for all previous tasks
	loss_dist=0
	for t_old in range(0,t):
		loss_dist += cross_entropy(outputs[t_old],targets_old[t_old],exp=1/T)

	# Cross entropy loss
	loss_ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=outputs[t]))
	loss = loss_ce+lamb*loss_dist

	return loss,loss_dist,loss_ce

def cross_entropy(outputs, targets, exp, eps=1e-5):
	out = tf.nn.softmax(outputs)
	tar = tf.nn.softmax(targets)
	out = tf.pow(out, exp)
	out = out / tf.expand_dims(tf.reduce_sum(out,axis=1),-1)
	tar = tf.pow(tar, exp)
	tar = tar / tf.expand_dims(tf.reduce_sum(tar, axis=1),-1)
	out = out + eps
	ce = -tf.reduce_mean(tf.reduce_sum(tar * tf.log(out), 1))

	return ce

def clone_net(net_old,net):
	for i in range(len(net.params)):
		assign_op = net_old.params[i].assign(net.params[i])
		sess.run(assign_op)



x = tf.placeholder(tf.float32,shape=[None,784])
model = Model(x,keep_prob=0.5)
# x_clone = tf.placeholder(tf.float32,shape=[None,784])
model_old = Model(x,keep_prob=0.5)
# train and test
num_tasks_to_run = 10
num_epochs_per_task = 50
minibatch_size = 256
learning_r = 0.1
epoches = 10*mnist.train.num_examples/minibatch_size #decay per 10 epoch
Lamb_set = [0.01,0.1,0.5,1,2,4,6,8]# lambda hyper-parameter search

# Generate the tasks specifications as a list of random permutations of the input pixels.
task_permutation = []
W_output = []
b_output = []
task_output = []
for task in range(num_tasks_to_run):
	# task_permutation.append( np.random.permutation(784) )
	mnist_ = permute_mnist(mnist,task)
	task_permutation.append(mnist_)

y_ = tf.placeholder(tf.float32, shape=[None, 10])
out_dim = int(y_.get_shape()[1]) # 10 for MNIST
lr_steps = tf.Variable(0)

for task in range(num_tasks_to_run):
	W = weight_variable([256,out_dim])
	b = bias_variable([out_dim])
	W_output.append(W)
	b_output.append(b)
for task in range(num_tasks_to_run):
	output = tf.matmul(model.h2_drop,W_output[task]) + b_output[task]
	task_output.append(output)

avg_performance = []

## Initialize session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

per_avg_performance = []
best_accuracy = 0.0

for Lamb in Lamb_set:
	# create and initialize sess
	sess = tf.InteractiveSession(config=config)
	sess.run(tf.global_variables_initializer())

	first_performance = []
	last_performance = []
	tasks_acc = []

	for task in range(num_tasks_to_run):
		# print "Training task: ",task+1,"/",num_tasks_to_run
		print("\t task ", task+1)

		if task == 0:
			cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=task_output[task]))
		else:
			T_target_old = []
			clone_net(model_old,model)
			for pre_t in range(task):
				T_target_old.append(tf.matmul(model_old.h2_drop,W_output[pre_t]) + b_output[pre_t])

			cost,cost_old,cost_new = lwf_criterion(t=task,targets_old=T_target_old,outputs=task_output,targets=y_,lamb=Lamb,T=2)

		train_op,learning_rate = optimizer(lr_start=learning_r, cost=cost, varlist=[model.params, W_output[task], b_output[task]],steps=lr_steps,epoches=epoches)
		sess.run(tf.variables_initializer([lr_steps]))
		for epoch in range(num_epochs_per_task):
			if epoch%5==0:
				print("\t Epoch ",epoch)
			for i in range(int(mnist.train.num_examples/minibatch_size)+1):
				batch = task_permutation[task].train.next_batch(minibatch_size)
				sess.run(train_op,feed_dict={x:batch[0], y_:batch[1]})
			# if task > 0:
			# 	print(sess.run(cost_new,feed_dict={x:batch[0], y_:batch[1]}))
			# print(sess.run(learning_rate))

		# Print test set accuracy to each task encountered so far
		avg_accuracy = 0.0
		for test_task in range(task+1):
			correct_prediction = tf.equal(tf.argmax(task_output[test_task], 1), tf.argmax(y_, 1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
			test_data = task_permutation[test_task].test

			acc = sess.run(accuracy, feed_dict={x:test_data.images, y_:test_data.labels}) * 100.0
			avg_accuracy += acc

			if test_task == 0:
				first_performance.append(acc)
			if test_task == task:
				last_performance.append(acc)

			tasks_acc.append(acc)
			print("Task: ",test_task," \tAccuracy: ",acc)

		avg_accuracy = avg_accuracy/(task+1)
		print("Avg Perf: ",avg_accuracy)

	if avg_accuracy > best_accuracy:
		best_accuracy = avg_accuracy
		best_acc = tasks_acc
	print('best-aver-acc ',best_accuracy)
	per_avg_performance.append(avg_accuracy)

best_lamb = Lamb_set[per_avg_performance.index(max(per_avg_performance))]
print('best lamb is: ',best_lamb)

# save best model-test log
file= open('log-lwf.txt', 'w')
for fp in best_acc:
	file.write(str(fp))
	file.write('\n')
file.write(str(per_avg_performance))
file.close()




sess.close()

## plot result
# tasks = range(1,num_tasks_to_run+1)
# plt.plot(tasks, first_performance)
# plt.plot(tasks, last_performance)
# plt.plot(tasks, avg_performance)
# plt.legend(["Task 0 (t=i)", "Task i (t=i)", "Avg Task (t=i)"], loc='lower right')
# plt.xlabel("Task")
# plt.ylabel("Accuracy (%)")
# plt.ylim([50, 100])
# plt.xticks(tasks)
# plt.show()
