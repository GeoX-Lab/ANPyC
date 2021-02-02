import tensorflow as tf
import numpy as np
from math import ceil
from copy import deepcopy
from tensorflow.examples.tutorials.mnist import input_data
# from util import *
import random
import matplotlib.pyplot as plt
import seaborn as sn



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
	def __init__(self, x,keep_prob=0.5):
		# create network 784-50-50-10
		in_dim = int(x.get_shape()[1]) # 784 for MNIST

		self.x = x # input placeholder

		# simple 3-layer network
		self.W1 = weight_variable([in_dim,64])
		self.b1 = bias_variable([64])
		self.W2 = weight_variable([64, 32])
		self.b2 = bias_variable([32])

		self.h1 = tf.nn.relu(tf.matmul(self.x, self.W1) + self.b1)  # hidden layer
		self.h1_drop = tf.nn.dropout(self.h1, keep_prob)

		self.h2 = tf.nn.relu(tf.matmul(self.h1_drop,self.W2) + self.b2) # hidden layer
		self.h2_drop = tf.nn.dropout(self.h2,keep_prob)
		self.params = [self.W1,self.b1,self.W2,self.b2]
		return

def compute_omega(sess,x,keep_prob, imgset, batch_size, param_im):
	Omega_M = []
	batch_num = ceil(len(imgset) / batch_size)
	for iter in range(batch_num):
		index = iter * batch_size
		Omega_M.append(sess.run(param_im, feed_dict={x: imgset[index:(index + batch_size - 1)],keep_prob:1}))
	Omega_M = np.sum(Omega_M, 0) / batch_num

	return Omega_M

#  FI -- calculate params importance
def Cal_importance(varlist,output,flags='entropy'):
    importance = []
    probs = tf.nn.softmax(output)

    for v in range(len(varlist)):
        if flags == 'entropy':
            gradients = tf.gradients(-tf.reduce_sum(probs * tf.log(probs)), varlist[v])# Gradient of the loss function for the current task
            importance.append(tf.reduce_mean((gradients * varlist[v] + 1 / 2 * tf.multiply(tf.square(varlist[v]), tf.square(gradients))),0))
        elif flags == 'MAS':
            gradients = tf.gradients(tf.nn.l2_loss(probs), varlist[v])
            importance.append(tf.reduce_mean(tf.abs(gradients), 0))
        elif flags == 'EWC':
            class_ind = tf.to_int32(tf.multinomial(tf.log(probs), 1)[0][0])
            im = tf.gradients(tf.log(probs[0, class_ind]), varlist[v])
            importance.append(tf.reduce_mean(tf.square(im), 0))
    return importance

def returen_grad_revise(masks, loss, var_list, lr_start, steps, epoches):
    learning_rate = tf.train.exponential_decay(lr_start, global_step=steps, decay_steps=epoches, decay_rate=0.9,
                                               staircase=True)  # lr decay=1
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    grads_net = optimizer.compute_gradients(loss, var_list[0])
    grads_output = optimizer.compute_gradients(loss, var_list[1:])

    grads_net1, grads_net2 = zip(*grads_net)
    grads_net_ = []
    # Omega = im_normalize(Omega, p)
    for i in range(len(masks)):
        grads_net_.append((masks[i] * grads_net1[i], grads_net2[i]))

    grads = grads_net_ + grads_output
    train_op = optimizer.apply_gradients(grads, global_step=steps)

    return train_op


def optimizer(lr_start,cost,varlist,steps,epoches):
	learning_rate = tf.train.exponential_decay(lr_start, global_step=steps, decay_steps=epoches, decay_rate=0.9,staircase=True) # lr decay=1
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost,global_step=steps, var_list=varlist)
	return train_step,learning_rate

"given a prune percent for layer wised"
# generate weight masks for layers
# def get_mask_matrix(Omega_v,percent):
#     mask = deepcopy(Omega_v)  ##掩膜矩阵
#     for mm in range(len(Omega_v)):
#         mask[mm][abs(mask[mm]) > 0] = 1
#         threshold=np.percentile(abs(Omega_v[mm]),percent)
#         print("阈值：",threshold)
#         mask[mm][abs(Omega_v[mm]) < threshold] = 0
#         # mask[mm][abs(mask[mm]) > threshold] = 1##不能等于阈值，例如当阈值为0时，如果等于阈值，那么之前剪掉的参数又变回1了，又可以更新了
#     return mask

def get_mask_matrix(Omega_v,percent):
    mask = deepcopy(Omega_v)  ##掩膜矩阵
    for mm in range(len(Omega_v)):
        mask[mm][abs(mask[mm]) > 0] = 1
        threshold=np.percentile(Omega_v[mm],percent)
        print("阈值：",threshold)
        mask[mm][Omega_v[mm] < threshold] = 0
        # mask[mm][abs(mask[mm]) > threshold] = 1##不能等于阈值，例如当阈值为0时，如果等于阈值，那么之前剪掉的参数又变回1了，又可以更新了
    return mask

# apply masks on parameters
def apply_prune_weights(params, mask):
    assign_op = {}
    for dd in range(len(params)):
        assign_op["%d" % dd] = params[dd].assign(tf.multiply(mask[dd], params[dd]))
    return assign_op

# apply masks on grads
def prune_grads(grads_vars, mask):
    for s, (g, v) in enumerate(grads_vars):
        if g is not None:  ##
            grads_vars[s] = (tf.multiply(mask[s], g), v)  # prune gradients
    return grads_vars

# apply gradient punish, base the
def grad_revise(masks,loss,var_list,lr_start,steps,epoches):
	learning_rate = tf.train.exponential_decay(lr_start, global_step=steps, decay_steps=epoches, decay_rate=0.9,
                                               staircase=True)  # lr decay=1
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	grads_net = optimizer.compute_gradients(loss,var_list[0])
	grads_output = optimizer.compute_gradients(loss,var_list[1:])

	grads_net1,grads_net2 = zip(*grads_net)
	grads_net_ = []
	# Omega = im_normalize(Omega, p)
	for i in range(len(masks)):
		grads_net_.append((masks[i]*grads_net1[i],grads_net2[i]))

	grads = grads_net_ + grads_output
	train_op = optimizer.apply_gradients(grads,global_step=steps)

	return train_op




mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])
keep_prob = tf.placeholder(tf.float32, shape=[])
model = Model(x,keep_prob)
model_old = Model(x)




# train and test
num_tasks_to_run = 1
num_epochs_per_task = 10
minibatch_size = 128
epoches = 10*mnist.train.num_examples/minibatch_size #decay per 10 epoch
learning_r = 0.1
num_epochs_retrain = 2   # setting the start of prune
prune_percent = 90


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
	W = weight_variable([32, out_dim])
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

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=task_output[0]))
train_op, learning_rate = optimizer(lr_start=learning_r, cost=cost,
									varlist=[model.params, W_output[task], b_output[task]], steps=lr_steps,
									epoches=epoches)

sess.run(tf.variables_initializer([lr_steps]))

for epoch in range(num_epochs_per_task):
	if epoch % 5 == 0:
		print("\t Epoch ", epoch)

	for i in range(int(mnist.train.num_examples / minibatch_size) + 1):
		batch = task_permutation[task].train.next_batch(minibatch_size)
		sess.run(train_op, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1})

test_task = 0
# for test_task in range(task + 1):
correct_prediction = tf.equal(tf.argmax(task_output[test_task], 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
test_data = task_permutation[test_task].test

print('acc is :', sess.run(accuracy, feed_dict={x: test_data.images, y_: test_data.labels, keep_prob: 1}) * 100.0)




def layerF(Om):
	im = []
	for i in range(int(len(Om)/2)):
		a = Om[2*i].flatten()
		b = Om[2*i+1].flatten()
		im.append(np.array(a.tolist() + b.tolist()))
	return im

def allF(Om):
	im = []
	for i in range(len(Om)):
		a = Om[i].flatten()
		# b = Om[i*2].flatten()
		im = im + a.tolist()
	return np.array(im)



def normalizer(OM,flags=False):
    if flags is True:
        OM = (OM - np.min(OM))/(np.max(OM)-np.min(OM))
    else:
        OM = OM
    return OM


"cal importance"
param_importance_ewc = Cal_importance(varlist=model.params, output=task_output[task], flags='EWC')
param_importance_mas = Cal_importance(varlist=model.params, output=task_output[task], flags='MAS')
param_importance_spp = Cal_importance(varlist=model.params, output=task_output[task], flags='entropy')


Omega_ewc = compute_omega(sess,x,keep_prob, task_permutation[task].train.images, batch_size=100,param_im=param_importance_ewc)
Omega_mas = compute_omega(sess,x,keep_prob, task_permutation[task].train.images, batch_size=100,param_im=param_importance_mas)
Omega_spp_ = compute_omega(sess,x,keep_prob, task_permutation[task].train.images, batch_size=100,param_im=param_importance_spp)
Omega_spp = [np.maximum(enum,0) for enum in Omega_spp_]




"prune training"
for prune_iter in range(0,prune_percent,5):
	print("current prune percent is", prune_iter)

	mask = get_mask_matrix(Omega_spp_,prune_iter)
	weights_prune = apply_prune_weights(model.params,mask)

	prune_params = sess.run(weights_prune)

	train_op = grad_revise(mask, loss=cost, var_list=[model.params, W_output[task], b_output[task]]
						   ,lr_start=learning_r, steps=lr_steps, epoches=epoches)

	for epoch in range(num_epochs_retrain):
		if epoch % 5 == 0:
			print("\t Epoch ", epoch)
		for i in range(int(mnist.train.num_examples / minibatch_size) + 1):
			batch = task_permutation[task].train.next_batch(minibatch_size)
			sess.run(train_op, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1})

			if i % 50 == 0:
				print('*****prune acc is *****:',sess.run(accuracy, feed_dict={x: test_data.images, y_: test_data.labels, keep_prob: 1}) * 100.0)

train_op, learning_rate = optimizer(lr_start=learning_r, cost=cost,
									varlist=[model.params, W_output[task], b_output[task]], steps=lr_steps,
									epoches=epoches)
param_importance_ewc = Cal_importance(varlist=model.params, output=task_output[task], flags='EWC')
param_importance_mas = Cal_importance(varlist=model.params, output=task_output[task], flags='MAS')
param_importance_spp = Cal_importance(varlist=model.params, output=task_output[task], flags='entropy')

Omega_ewc1 = compute_omega(sess,x,keep_prob, task_permutation[task].train.images, batch_size=100,param_im=param_importance_ewc)
Omega_mas1 = compute_omega(sess,x,keep_prob, task_permutation[task].train.images, batch_size=100,param_im=param_importance_mas)
Omega_spp_1 = compute_omega(sess,x,keep_prob, task_permutation[task].train.images, batch_size=100,param_im=param_importance_spp)
Omega_spp1 = [np.maximum(enum,0) for enum in Omega_spp_1]


Omega_spp1 = [np.maximum(enum,0) for enum in Omega_spp_1]
Omega_layer_ewc = layerF(Omega_ewc)
Omega_layer_mas = layerF(Omega_mas)
Omega_layer_spp = layerF(Omega_spp)
Omega_layer_spp1 = layerF(Omega_spp1)


Omega_all_ewc = normalizer(allF(Omega_ewc),True)
Omega_all_mas = normalizer(allF(Omega_mas),True)
Omega_all_spp = normalizer(allF(Omega_spp),True)


Omega_all_ewc1 = normalizer(allF(Omega_ewc1),True)
Omega_all_mas1 = normalizer(allF(Omega_mas1),True)
Omega_all_spp1 = normalizer(allF(Omega_spp1),True)

def static_zeros(list):
	return sum([int(i==0) for i in list])

zero_ewc = static_zeros(Omega_all_ewc)
zero_mas = static_zeros(Omega_all_mas)
zero_spp = static_zeros(Omega_all_spp)

zero_ewc1 = static_zeros(Omega_all_ewc1)
zero_mas1 = static_zeros(Omega_all_mas1)
zero_spp1 = static_zeros(Omega_all_spp1)






# cal skeweness

import stats as sts
list_Omega = [Omega_all_ewc,Omega_all_mas,Omega_all_spp,Omega_all_ewc1,Omega_all_mas1, Omega_all_spp1]

list_skewness = []
for i in range(len(list_Omega)):
    list_skewness.append(sts.skewness(list_Omega[i])) 
    
# cal kurtosis
list_kurtosis = []
for i in range(len(list_Omega)):
    list_kurtosis.append(sts.kurtosis(list_Omega[i]))

plt.figure()
sn.distplot(Omega_all_ewc, rug=True,bins=1000,hist=False, kde=True, kde_kws={'bw':0.01}, label='ewc')
sn.distplot(Omega_all_mas, rug=True,bins=1000,hist=False, kde=True, kde_kws={'bw':0.01}, label='mas')
sn.distplot(Omega_all_spp, rug=True,bins=1000,hist=False, kde=True, kde_kws={'bw':0.01}, label='ours')
plt.legend(prop={'size': 16})
# plt.title('Density with param-importance')
plt.xlabel('importance',fontsize=16)
plt.ylabel('Density',fontsize=16)
plt.savefig('overall1-1.jpg',dpi=300)
plt.show()


plt.figure()
sn.distplot(Omega_all_spp, rug=True, bins=1000,hist=False,color='g', kde=True, kde_kws={'bw':0.01}, label='ours')
sn.distplot(Omega_all_spp1,rug=True, bins=1000,hist=False,color='r', kde=True, kde_kws={'bw':0.01}, label='ours_np')
plt.legend(prop={'size': 16})
# plt.title('Density with param-importance')
plt.xlabel('importance',fontsize=16)
plt.ylabel('Density',fontsize=16)
plt.savefig('overall1-2.jpg',dpi=300)
plt.show()



plt.figure()
sn.distplot(Omega_all_ewc, rug=True,bins=1000,hist=False,color='g', kde=True, kde_kws={'bw':0.01}, label='ewc')
sn.distplot(Omega_all_ewc1, rug=True,bins=1000,hist=False,color='r', kde=True, kde_kws={'bw':0.01}, label='ewc_np')
plt.legend(prop={'size': 16})
# plt.title('Density with param-importance')
plt.xlabel('importance',fontsize=16)
plt.ylabel('Density',fontsize=16)
plt.savefig('overall1-3.jpg',dpi=300)
plt.show()


plt.figure()
sn.distplot(Omega_all_mas, rug=True,bins=1000,hist=False,color='g', kde=True, kde_kws={'bw':0.01}, label='mas')
sn.distplot(Omega_all_spp1, rug=True,bins=1000,hist=False,color='r', kde=True, kde_kws={'bw':0.01}, label='mas_np')
# sn.kdeplot(Omega_all_mas,shade=True,bw=0.1,color="g",label='mas')
# sn.kdeplot(Omega_all_mas1,shade=True,bw=0.1,color="b",label='mas_np')
plt.legend(prop={'size': 16})
# plt.title('Density with param-importance')
plt.xlabel('importance',fontsize=16)
plt.ylabel('Density',fontsize=16)
plt.savefig('overall1-4.jpg',dpi=300)
plt.show()




# 峰度、偏度
a1=[221.43370004395413,
24.348012507670884,
1919.2857036814637,
623.2910328785347,
48.240862162789405,
14052.965363228466]

b1=[11.143564067779455,
2.980284057749085,
33.10979700611376,
21.29774556412941,
4.869924745736586,
106.13443147471386]

###
a2=[245.05033874629603,
9.630427489455236,
750.2008327670181,
325.5976588392647,
34.528876023541834,
6796.092692582839]


b2=[11.753379727127157,
2.309149592180777,
21.007722015278727,
14.935567332916609,
4.358692884239893,
75.72474187262624]

### 

a3=[316.1855058959434,
11.907416484433606,
4112.270793221261,
2346.5472926709735,
65.99990451428684,
39639.98272460187]


b3=[12.434560721045077,
2.383879360023878,
45.77357876259365,
39.65505215878588,
5.266472417703068,
189.89253069019637]


###

a4=[112.69411800072295,
6.445490829837087,
810.2812790382151,
607.9223879621943,
37.196357447598224,
8937.650177060692]

b4=[7.888504998977062,
1.8999063539929846,
19.64480900944239,
20.841984210393584,
4.302009433467407,
87.50579071081229]

###

a5=[227.768435946527,
8.427519374017184,
598.7271435689098,
1266.1649342331539,
47.141480702520155,
28759.24256731111]

b5=[10.855649661651807,
2.135997926701109,
18.620739177718267,
28.583644169584534,
4.821617511623856,
156.17356270474397]

###
a6=[223.7193427320243,
8.769274230392636,
1245.9692365197398,
533.7333035307107,
39.60664050584674,
22732.961068203374]

b6=[11.296204730328233,
2.2392586735958093,
25.057772789347865,
18.949493951328567,
4.737028755898035,
138.23997612638425]

###

a7=[697.9610483535839,
10.12532225540753,
1713.653940243816,
499.2410759508862,
39.62639291606203,
8801.87569435327]

b7=[20.09886596804014,
2.2077111284243487,
29.61120412869746,
18.59980361690498,
4.309044616429855,
89.75386369911335]

###
a8=[617.0090532540361,
22.23549689177933,
2925.04868377694,
639.0303463208231,
61.00753027606628,
7566.89018788806]

b8=[15.641255522618016,
2.852678343892391,
40.09229038682504,
20.360046616276097,
5.4347716290863675,
80.53462511269537]

###
a9=[177.67558547205815,
7.7655051493231255,
760.7812682193627,
600.8947141836347,
37.35186515656828,
7386.177107171987]

b9=[10.648964636220786,
2.0678954086069363,
20.777078006236135,
20.64893093999057,
4.181051060238329,
80.86717049385334]

###
a10 = [680.2809700236319,
9.200683811011652,
1124.9476191982478,
488.1817664798129,
79.55962233366105,
6842.551873376791]

b10 = [19.15379162729723,
2.2180822364846615,
24.120352478338244,
20.00808455102135,
6.481168294321419,
79.22362763850137]


a = [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10]
b = [b1,b2,b3,b4,b5,b6,b7,b8,b9,b10]
array_a = np.array(a[0])
for i in range(9):
    array_a = np.vstack((array_a,np.array(a[i+1])))
    
array_b = np.array(b[0])
for i in range(9):
    array_b = np.vstack((array_b,np.array(b[i+1])))

# sum
mean_a = (np.array(a1) + np.array(a2)+ np.array(a3)+ np.array(a4)+ np.array(a5)+ np.array(a6)+ np.array(a7)+ np.array(a8)+ np.array(a9)+ np.array(a10))/10

mean_b = (np.array(b1) + np.array(b2)+ np.array(b3)+ np.array(b4)+ np.array(b5)+ np.array(b6)+ np.array(b7)+ np.array(b8)+ np.array(b9)+ np.array(b10))/10

mean_a = np.mean(array_a,0)
mean_b = np.mean(array_b,0)

std_a = np.std(array_a,0)
std_b = np.std(array_b,0)

labels = ['ewc', 'mas', 'connectivity']
a_means = [351.978,11.8855,1596.12]
a_means_np = [793.06,49.026,15151.6]

a_std = [211.424,5.88273,1076.37]
a_std_np = [567.975,14.2535,10858.9]



# width = 0.25       # the width of the bars: can also be len(x) sequence
y_pos = np.arange(3)
total_width, n = 0.6, 2
width = total_width / n
y_pos = y_pos - (total_width - width) / 2

fig, ax = plt.subplots(2,1)

ax1 = ax[0]
ax2 = ax[1]

ax1.barh(y_pos, a_means, xerr=a_std, height=width,alpha=0.5,align='center',label='orignal')
ax1.barh(y_pos+width, a_means_np,xerr=a_std_np,height=width,alpha=0.5, align='center',label='neural prune')
ax1.set_yticks(y_pos)
ax1.set_yticklabels(labels)
ax1.invert_yaxis()  # labels read top-to-bottom
ax1.set_xlabel('kurtosis')
ax1.legend()




b_means = [13.0915,2.32948,27.7815]
b_means_np = [22.388,4.87618,108.405]

b_std = [3.73673,0.320379,8.79149]

b_std_np = [6.59264,0.668209,37.4922]




ax2.barh(y_pos, b_means, xerr=b_std, height=width,alpha=0.5,align='center',label='orignal')
ax2.barh(y_pos+width, b_means_np, xerr=b_std_np, height=width,alpha=0.5,align='center',label='neural prune')
ax2.set_yticks(y_pos)
ax2.set_yticklabels(labels)
ax2.invert_yaxis()  # labels read top-to-bottom
ax2.set_xlabel('skewness')
ax2.legend()

plt.tight_layout()
plt.savefig('kurtosis and skeweness.png',dpi=300)
plt.show()


labels = ['ewc', 'mas', 'ours']
a_means = [351.978,118.855,1596.12]
a_means_np = [793.06,490.26,15151.6]

a_std = [211.424,58.8273,1076.37]
a_std_np = [567.975,142.535,10858.9]



# width = 0.25       # the width of the bars: can also be len(x) sequence
x_pos = np.arange(3)
width = 0.3
# total_width, n = 0.4, 2
# width = total_width / n
# x_pos = x_pos - (total_width - width) / 2
# plt.figure(figsize=(8, 15), dpi=300)
fig, ax = plt.subplots(1,2,figsize=(7.5, 9))

ax1 = ax[0]
ax2 = ax[1]

ax1.bar(x_pos, a_means, yerr=a_std, width=width,alpha=0.8,align='center',label='orignal')
ax1.bar(x_pos+width, a_means_np,yerr=a_std_np, width=width,alpha=0.8, align='center',label='neural prune')
ax1.set_xticks(x_pos)
# ax1.set_xticklabels(labels,fontsize=16,rotation=60)
ax1.set_xticklabels(labels,fontsize=18)
# ax1.invert_xaxis()  # labels read top-to-bottom
ax1.set_ylabel('kurtosis',fontsize=18)
ax1.legend(prop={'size': 15},loc='best')




b_means = [13.0915,2.32948,27.7815]
b_means_np = [22.388,4.87618,108.405]

b_std = [3.73673,0.320379,8.79149]

b_std_np = [6.59264,0.668209,37.4922]




ax2.bar(x_pos, b_means, yerr=b_std, width=width,alpha=0.8,align='center',label='orignal')
ax2.bar(x_pos+width, b_means_np, yerr=b_std_np, width=width,alpha=0.8,align='center',label='neural prune')
ax2.set_xticks(x_pos)
# ax2.set_xticklabels(labels,fontsize=16,rotation=60)
ax2.set_xticklabels(labels,fontsize=18)
# ax2.invert_xaxis()  # labels read top-to-bottom
ax2.set_ylabel('skewness',fontsize=18)
ax2.legend(prop={'size': 15},loc='best')

plt.tight_layout()
plt.savefig('kurtosis and skeweness.png',dpi=300)
plt.show()

