import tensorflow as tf
import numpy as np
from math import ceil
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

def compute_omega(sess, imgset, batch_size, param_im,sharpen=False,beta=2):
	Omega_M = []
	batch_num = ceil(len(imgset) / batch_size)
	for iter in range(batch_num):
		index = iter * batch_size
		Omega_M.append(sess.run(param_im, feed_dict={x: imgset[index:(index + batch_size - 1)]}))
	Omega_M = np.sum(Omega_M, 0) / batch_num
	if sharpen:
		Omega_M = im_sharpen(Omega_M,beta)
	return Omega_M

# loss
# def compute_omega(sess, imgset, batch_size,label, param_im,shapren=True,beta=2):
# 	Omega_M = []
# 	batch_num = ceil(len(imgset) / batch_size)
# 	for iter in range(batch_num):
# 		index = iter * batch_size
# 		Omega_M.append(sess.run(param_im, feed_dict={x: imgset[index:(index + batch_size - 1)],y_: label[index:(index + batch_size - 1)]}))
# 	Omega_M = np.sum(Omega_M, 0) / batch_num
#
# # importance sharpen
# # 	if shapren:
# # 		Omega_M = im_sharpen(Omega_M,beta)
# #
# # 	return Omega_M


def optimizer(lr_start,cost,varlist,steps,epoches):
	learning_rate = tf.train.exponential_decay(lr_start, global_step=steps, decay_steps=epoches, decay_rate=0.9,staircase=True) # lr decay=1
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost,global_step=steps, var_list=varlist)
	return train_step,learning_rate

# calculate params importance
def Cal_importance(varlist,output):
	importance = []
	probs = tf.nn.softmax(output)
	for v in range(len(varlist)):
		# # information entropy
		I = -tf.reduce_sum(probs * tf.log(probs))

		# # l2
		# I = tf.reduce_sum(tf.square(probs),1)

		gradients = tf.gradients(I, varlist[v])# Gradient of the loss function for the current task
		# cal importance --abs
		# importance.append(tf.reduce_mean(
		# 	tf.abs(gradients * varlist[v] + 1 / 2 * tf.multiply(tf.square(varlist[v]), tf.square(gradients))), 0))

	# # cal importance --max（0,a）
		importance.append(tf.reduce_mean(
			tf.maximum((gradients * varlist[v] + 1 / 2 * tf.multiply(tf.square(varlist[v]), tf.square(gradients))), 0),0))

	# 	# MAS
	# 	importance.append(tf.reduce_mean(tf.abs(gradients), 0))

	# 	# cal importance -- gradient based;max（0,im)
	# 	importance.append(tf.reduce_mean(
	# 		tf.maximum(gradients * varlist[v],0), 0))
	return importance

# # loss-importance based
# def Cal_importance(varlist, output,y_tgt):
# 	importance = []
# 	probs = tf.nn.softmax(output)
# 	for v in range(len(varlist)):
# 		# loss
# 		I = -tf.reduce_sum(y_tgt * tf.log(probs + 1e-04) + (1. - y_tgt) * tf.log(1. - probs + 1e-04))
# 		gradients = tf.gradients(I, varlist[v])  # Gradient of the loss function for the current task
#
# 	# cal importance --max（0,a）
# 		importance.append(tf.reduce_mean(
# 			tf.maximum((gradients * varlist[v] + 1 / 2 * tf.multiply(tf.square(varlist[v]), tf.square(gradients))), 0),0))
#
# 	return importance

# def im_sharpen(im_value,sharpen=True,beta=1):
# 	O_im = []
# 	if sharpen:
# 		total = 0
# 		for j in im_value:
# 			total += np.sum(np.exp(beta * j))
# 		for i in im_value:
# 			temp = np.exp(beta * i) / total
# 			O_im.append(temp)
# 	else:
# 		O_im = im_value
# 	return O_im

def im_sharpen(im_value,beta=1):
	O_im = []
	# total = 0
	# for j in im_value:
	# 	total += np.sum(np.exp(beta * j))
	# for i in im_value:
	# 	temp = np.exp(beta * i) / total
	# 	O_im.append(temp)
	for i in im_value:
		temp = np.exp(beta * i)
		O_im.append(temp)
	return O_im

# update loss function, with auxiliary loss
def update_loss(pre_var,Omega,var,new_loss,lamb):
	aux_loss = 0
	for v in range(len(pre_var)):
		aux_loss += tf.reduce_sum(tf.multiply(Omega[v], tf.square(pre_var[v] - var[v])))
	loss = new_loss + lamb * aux_loss
	return loss


def clone_net(net_old, net):
	for i in range(len(net.params)):
		assign_op = net_old.params[i].assign(net.params[i])
		sess.run(assign_op)

# apply gradient punish, base the
# def gra_punish(Omega):
# 	gradients_with_aux = optimizer.compute_gradients(cross_entropy + param_c * aux_loss, var_list=variables)
#
# 	for i, (grad, var) in enumerate(gradients_with_aux):
# 		update_small_omega_ops.append(tf.assign_add(small_omega_var[var.op.name],
# 													learning_rate * gradients_with_aux[i][0] * gradients[i][
# 														0]))  # small_omega -= delta_weight(t)*gradient(t)
#
# 	update_small_omega = tf.group(*update_small_omega_ops)  # 1) update small_omega after each train!
#
# 	train = optimizer.apply_gradients(gradients_with_aux)

mnist = input_data.read_data_sets("/tmp/", one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])
model = Model(x)
# x_clone = tf.placeholder(tf.float32,shape=[None,784])
model_old = Model(x)
# train and test
num_tasks_to_run = 10
num_epochs_per_task = 50
minibatch_size = 256
Lamb_set = [0.5,1.5,2,2.5,3,4,6,7,8,10,12,15]# hyper-parameter1 loss
epoches = 10*mnist.train.num_examples/minibatch_size #decay per 10 epoch
Beta = 0.001 # hyper-parameter2 sharpen
learning_r = 0.1

# Generate the tasks specifications as a list of random permutations of the input pixels.
task_permutation = []
for task in range(num_tasks_to_run):
	# task_permutation.append( np.random.permutation(784) )
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


per_avg_performance = []
best_accuracy = 0.0

for Lamb in Lamb_set:
	sess = tf.InteractiveSession(config=config)
	sess.run(tf.global_variables_initializer())

	Omega_v = []
	first_performance = []
	last_performance = []
	tasks_acc = []

	for task in range(num_tasks_to_run):
		# print "Training task: ",task+1,"/",num_tasks_to_run
		print("\t task ", task + 1)

		if task == 0:
			cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=task_output[task]))
		else:
			# Lamb_ = 2/3*Lamb*((num_tasks_to_run-task)/num_tasks_to_run)+1/3*Lamb
			clone_net(model_old, model)
			cost_new = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=task_output[task]))
			cost = update_loss(pre_var=model_old.params, Omega=Omega_v, var=model.params, new_loss=cost_new, lamb=Lamb)
		# sgd
		# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=task_output[task]))
		# train_op = optimizer(lr_start=learning_r, cost=cost, varlist=[model.params, W_output[task], b_output[task]])
		train_op,learning_rate = optimizer(lr_start=learning_r, cost=cost, varlist=[model.params, W_output[task], b_output[task]],steps=lr_steps,epoches=epoches)
		sess.run(tf.variables_initializer([lr_steps]))

		for epoch in range(num_epochs_per_task):
			if epoch % 5 == 0:
				print("\t Epoch ", epoch)

			for i in range(int(mnist.train.num_examples / minibatch_size) + 1):
				batch = task_permutation[task].train.next_batch(minibatch_size)
				sess.run(train_op, feed_dict={x: batch[0], y_: batch[1]})
		# calculate params importance
		param_importance = Cal_importance(varlist=model.params, output=task_output[task])

		# param_importance = Cal_importance(varlist=model.params, output=task_output[task],y_tgt=y_) # loss
		if task == 0:
			Omega_v = compute_omega(sess, task_permutation[task].train.images, batch_size=100,param_im=param_importance,sharpen=False,beta=Beta)
			print(np.max(Omega_v[0]))
		# Omega_v = compute_omega(sess, task_permutation[task].train.images, 100, task_permutation[task].train.labels, param_im=param_importance)
		else:
			for l in range(len(Omega_v)):
				# loss
				# Omega_v[l] = Omega_v[l] + compute_omega(sess, task_permutation[task].train.images, 100, task_permutation[task].train.labels, param_im=param_importance)[l]
				# Omega_v[l] = Omega_v[l] + compute_omega(sess, task_permutation[task].train.images, batch_size=100, param_im=param_importance,sharpen=False,beta=Beta)[l]
				Omega_v[l] = np.maximum(Omega_v[l],compute_omega(sess, task_permutation[task].train.images, batch_size=100, param_im=param_importance,sharpen=False,beta=Beta)[l])
			# sharpen
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


	if avg_accuracy > best_accuracy:
		best_accuracy = avg_accuracy
		best_acc = tasks_acc
	print('best-aver-acc ',best_accuracy)
	per_avg_performance.append(avg_accuracy)

best_lamb = Lamb_set[per_avg_performance.index(max(per_avg_performance))]
print('best lamb is: ',best_lamb)

# save best model-test log
file= open('log-oursharpen.txt', 'w')
for fp in best_acc:
	file.write(str(fp))
	file.write('\n')
file.write(str(per_avg_performance))
file.close()



sess.close()
# plot results
# tasks = range(1, num_tasks_to_run + 1)
# plt.plot(tasks, first_performance)
# plt.plot(tasks, last_performance)
# plt.plot(tasks, avg_performance)
# plt.legend(["Task 0 (t=i)", "Task i (t=i)", "Avg Task (t=i)"], loc='lower right')
# plt.xlabel("Task")
# plt.ylabel("Accuracy (%)")
# plt.ylim([50, 100])
# plt.xticks(tasks)
# plt.show()
