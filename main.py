from __future__ import print_function
from gazeboworld import GazeboWorld
import csv
import tensorflow as tf
import random
import numpy as np
import time
import rospy
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import os
NUM_PARALLEL_EXEC_UNITS = 6
#config = tf.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS, inter_op_parallelism_threads=2,
                       #allow_soft_placement=True, device_count={'CPU': NUM_PARALLEL_EXEC_UNITS},gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.80))

os.environ["OMP_NUM_THREADS"] = "6"
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from collections import deque
import keras as k
filename = "logs_reward.csv"
reward_file = open(filename, "w")
reward_writer=csv.writer(reward_file, delimiter=',')
reward_writer.writerow(['Step', 'Reward', 'Loss','value'])
filename1="succes_reward.csv"
succ_file=open(filename1,"w")
succ_writer=csv.writer(succ_file,delimiter=',')
succ_writer.writerow(['Episode','success rate'])

GAME = 'GazeboWorld'
ACTIONS = 7 # number of valid actions
SPEED = 2 # DoF of speed
GAMMA = 0.90 # decay rate of past observations
OBSERVE = 5. # timesteps to observe before training
EXPLORE = 200. # frames over which to anneal epsilon
FINAL_EPSILON = 0.001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 5000 # number of previous transitions to remember
BATCH = 64 # size of minibatch
MAX_EPISODE = 1211
MAX_T = 200
DEPTH_IMAGE_WIDTH = 160
DEPTH_IMAGE_HEIGHT = 128
RGB_IMAGE_HEIGHT = 228
RGB_IMAGE_WIDTH = 304
CHANNEL = 3
TAU = 0.001 # Rate to update target network toward primary network
H_SIZE = 8*10*64
IMAGE_HIST = 4


def variable_summaries(var):
	"""Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
	with tf.name_scope('summaries'):
		mean = tf.reduce_mean(var)
	tf.summary.scalar('mean', mean)
	with tf.name_scope('stddev'):
		stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
	tf.summary.scalar('stddev', stddev)
	tf.summary.scalar('max', tf.reduce_max(var))
	tf.summary.scalar('min', tf.reduce_min(var))
	tf.summary.histogram('histogram', var)

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.01)
	return tf.Variable(initial, name="weights")

def bias_variable(shape):
	initial = tf.constant(0., shape = shape)
	return tf.Variable(initial, name="bias")

def conv2d(x, W, stride_h, stride_w):
	return tf.nn.conv2d(x, W, strides = [1, stride_h, stride_w, 1], padding = "SAME")

def layer_norm(x,y):
       return tf.contrib.layers.layer_norm(x,activation_fn=y, trainable=True, begin_norm_axis=1, begin_params_axis=-1, center=True, scale=True)



class QNetwork(object):
	"""docstring for ClassName"""
	def __init__(self,sess):
		# network weights
		# input 80x100x4
		with tf.name_scope("Conv1"):
			W_conv1 = weight_variable([10, 14, 4, 32])
			variable_summaries(W_conv1)
			b_conv1 = bias_variable([32])
		# 8x12x16 # 20x25x16
		with tf.name_scope("Conv2"):
			W_conv2 = weight_variable([4, 4, 32, 64])
			variable_summaries(W_conv2)
			b_conv2 = bias_variable([64])
		# 4x4x32 # 10x13x32
		with tf.name_scope("Conv3"):
			W_conv3 = weight_variable([3, 3, 64, 64])
			variable_summaries(W_conv3)
			b_conv3 = bias_variable([64])
		# 3x3x64 #10x13x32
		with tf.name_scope("FCValue"):
			W_value = weight_variable([H_SIZE, 512])
			variable_summaries(W_value)
			b_value = bias_variable([512])
			# variable_summaries(b_ob_value)

		with tf.name_scope("FCAdv"):
			W_adv = weight_variable([H_SIZE, 512])
			variable_summaries(W_adv)
			b_adv = bias_variable([512])
			# variable_summaries(b_adv)


		# input layer
		self.state = tf.placeholder("float", [None, DEPTH_IMAGE_HEIGHT, DEPTH_IMAGE_WIDTH, IMAGE_HIST])
		# Conv1 layer
		h_conv_ln=layer_norm(self.state,tf.identity)
		h_conv1 = conv2d(h_conv_ln, W_conv1, 8, 8) + b_conv1
		#h_conv1 = tf.nn.max_pool(h_conv1, ksize=4, strides=1, padding="SAME")
		conv1 = tf.nn.leaky_relu(h_conv1, alpha=0.2)
		# Conv2 layer
		h_conv_ln1=layer_norm(conv1,tf.nn.relu)
		h_conv2 = conv2d(h_conv_ln1, W_conv2, 2, 2) + b_conv2
		#h_conv2 = tf.nn.max_pool(h_conv2, ksize=4,strides=1, padding="SAME")
		conv2 = tf.nn.leaky_relu(h_conv2, alpha=0.2)
		# Conv2 layer
		h_conv_ln2=layer_norm(conv2,tf.nn.relu)
		h_conv3 = conv2d(h_conv_ln2, W_conv3, 1, 1) + b_conv3
		#h_conv3 = tf.nn.max_pool(h_conv3, ksize=4,strides=1, padding="SAME")
		conv3 = tf.nn.leaky_relu(h_conv3, alpha=0.2)
		h_conv3_flat = tf.reshape(conv3, [-1, H_SIZE])
		
		value_out=self.noisy_dense(h_conv3_flat,input_size=5120, output_size=512,activation_fn=tf.nn.relu)
		value = self.noisy_dense(value_out,input_size=512, output_size=1)

		Adv_out=self.noisy_dense(h_conv3_flat,input_size=5120, output_size=512,activation_fn=tf.nn.relu)
		advantage=self.noisy_dense(Adv_out,input_size=512, output_size=ACTIONS)

		# Q = value + (adv - advAvg)
		advAvg = tf.expand_dims(tf.reduce_mean(advantage, axis=1), axis=1)
		advIdentifiable = tf.subtract(advantage, advAvg)
		self.readout = tf.add(value, advIdentifiable)
		self.readout1 = tf.reduce_mean(self.readout)
		#self.readout = tf.nn.leaky_relu(readout1, alpha=0.2)

		# define the ob cost function'
		self.a = tf.placeholder("float", [None, ACTIONS])
		self.y = tf.placeholder("float", [None])
		self.readout_action = tf.reduce_sum(tf.multiply(self.readout,self.a), axis=1)
		self.td_error = tf.square(self.y - self.readout_action)
		self.cost = tf.reduce_mean(self.td_error)
		self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cost)

	def noisy_dense(self,x,input_size,output_size, activation_fn=tf.identity):
                def f(x):
                        return tf.multiply(tf.sign(x), tf.pow(tf.abs(x), 0.5))
                sigma_init = (0.4/np.power(input_size, 0.5))
                p = tf.random_normal([input_size, 1])
                q = tf.random_normal([1, output_size])
                f_p = f(p)
                f_q = f(q)

                w_epsilon = f_p*f_q
                b_epsilon = tf.squeeze(f_q)
                w_mu = tf.Variable(tf.random_uniform([input_size, output_size],-1*1/np.power(input_size, 0.5),1*1/np.power(input_size, 0.5)))
                w_sigma = tf.Variable(tf.truncated_normal([input_size, output_size],stddev=sigma_init))
                w = w_mu + tf.multiply(w_sigma, w_epsilon)

                ret = tf.matmul(x, w)
                b_mu = tf.Variable(tf.random_uniform([output_size],-1*1/np.power(input_size, 0.5),1*1/np.power(input_size, 0.5)))
                b_sigma = tf.Variable(tf.truncated_normal([output_size],stddev=sigma_init))
                b = b_mu + tf.multiply(b_sigma, b_epsilon)
                return activation_fn(ret + b)


def updateTargetGraph(tfVars,tau):
	total_vars = len(tfVars)
	op_holder = []
	for idx,var in enumerate(tfVars[0:total_vars/2]):
		op_holder.append(tfVars[idx+total_vars/2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars/2].value())))
	return op_holder

def updateTarget(op_holder,sess):
	for op in op_holder:
		sess.run(op)

def trainNetwork():
	#sess = tf.InteractiveSession(config=config)
	config=tf.ConfigProto()
	config.gpu_options.allow_growth = True
	#config=config.gpu_options.per_process_gpu_memory_fraction=0.75
	#sess=K.tensorflow_backend.set_session(tf.Session(config=config))
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.65)
	#sess = tf.InteractiveSession(config=config)
	sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options,intra_op_parallelism_threads=4, inter_op_parallelism_threads=4,device_count={'CPU': 4}))
	with tf.name_scope("OnlineNetwork"):
		online_net = QNetwork(sess)
	with tf.name_scope("TargetNetwork"):
		target_net = QNetwork(sess)
	rospy.sleep(1.)
	reward_var = tf.Variable(0., trainable=False)
	reward_epi = tf.summary.scalar('reward', reward_var)
	loss_var = tf.Variable(0., trainable=False)
	loss_epi = tf.summary.scalar('loss', loss_var)
	value_var = tf.Variable(0., trainable=False)
	value_epi = tf.summary.scalar('value', value_var)
	# define summary
	merged_summary = tf.summary.merge_all()
	summary_writer = tf.summary.FileWriter('./logs_reward/', sess.graph)

	# Initialize the World
	env = GazeboWorld()
	print('Environment initialized')

	# Initialize the buffer
	D = deque()

	# get the first state
	depth_img_t1 = env.GetDepthImageObservation()
	depth_imgs_t1 = np.stack((depth_img_t1, depth_img_t1, depth_img_t1, depth_img_t1), axis=2)
	terminal = False

	# saving and loading networks
	trainables = tf.trainable_variables()
	trainable_saver = tf.train.Saver(trainables, max_to_keep=1)
	sess.run(tf.global_variables_initializer())
	checkpoint = tf.train.get_checkpoint_state("./saved_networks/")
	print('checkpoint:', checkpoint)
	if checkpoint and checkpoint.model_checkpoint_path:
		trainable_saver.restore(sess, checkpoint.model_checkpoint_path)
		print("Successfully loaded:", checkpoint.model_checkpoint_path)
	else:
		print("Could not find old network weights")

	# start training
	episode = 0

	r_epi = 0.
	t = 0
	T = 0
	rate = rospy.Rate(5)
	global BATCH
	global GAMMA
	print('Number of trainable variables:', len(trainables))
	targetOps = updateTargetGraph(trainables,TAU)
	loop_time = time.time()
	last_loop_time = loop_time
	while episode < MAX_EPISODE:
		env.ResetWorld()
		t = 0
		r_epi = 0.
		loss=0
		value=0
		terminal = False
		reset = False
		loop_time_buf = []
		action_index = 0
		while not reset and not rospy.is_shutdown():
			depth_img_t1 = env.GetDepthImageObservation()
			depth_img_t1 = np.reshape(depth_img_t1, (DEPTH_IMAGE_HEIGHT, DEPTH_IMAGE_WIDTH, 1))
			depth_imgs_t1 = np.append(depth_img_t1, depth_imgs_t1[:, :, :(IMAGE_HIST - 1)], axis=2)
			reward_t, terminal, count, reset = env.GetRewardAndTerminate(t)
			if t > 0 :
				D.append((depth_imgs_t, a_t, reward_t, depth_imgs_t1, terminal))
				if len(D) > REPLAY_MEMORY:
					del(D[0])
			depth_imgs_t = depth_imgs_t1

			# choose an action epsilon greedily
			a = sess.run(online_net.readout, feed_dict = {online_net.state : [depth_imgs_t1]})
			readout_t = a[0]
			a_t = np.zeros([ACTIONS])
			action_index=np.argmax(readout_t)
			a_t[action_index]=1

			# Control the agent
			env.Control(action_index)



			if episode > OBSERVE:
				# # sample a minibatch to train on
				
				minibatch = random.sample(D, BATCH)
				y_batch = []
				# get the batch variables
				depth_imgs_t_batch = [d[0] for d in minibatch]
				#print("State:",depth_imgs_t_batch)
				a_batch = [d[1] for d in minibatch]
				r_batch = [d[2] for d in minibatch]
				depth_imgs_t1_batch = [d[3] for d in minibatch]
				Q1 = online_net.readout.eval(feed_dict = {online_net.state : depth_imgs_t1_batch})
				Q2 = target_net.readout.eval(feed_dict = {target_net.state : depth_imgs_t1_batch})
				for i in range(0, len(minibatch)):
					terminal_batch = minibatch[i][4]
					# if terminal, only equals reward
					if terminal_batch:
						y_batch.append(r_batch[i])
					else:
						y_batch.append(r_batch[i] + GAMMA * Q2[i, np.argmax(Q1[i])])

				#Update the network with our target values.
				online_net.train_step.run(feed_dict={online_net.y : y_batch,
													online_net.a : a_batch,
													online_net.state : depth_imgs_t_batch })

				loss = sess.run(online_net.cost, feed_dict={online_net.y: y_batch, online_net.a : a_batch,online_net.state : depth_imgs_t_batch })
				value = sess.run(online_net.readout1, feed_dict={online_net.y: y_batch, online_net.a : a_batch,online_net.state : depth_imgs_t_batch})

				updateTarget(targetOps, sess) # Set the target network to be equal to the primary network.
				if (episode+1)%100 == 0:
                                        GAMMA = GAMMA + 0.01
                                if GAMMA >= 0.99:
                                        GAMMA = 0.99
 

			r_epi = r_epi + reward_t
                        reward_writer.writerow([T, r_epi,loss,value])
                        reward_file.flush()
			t += 1
			T += 1
			last_loop_time = loop_time
			loop_time = time.time()
			loop_time_buf.append(loop_time - last_loop_time)
			rate.sleep()
			


		#  write summaries
		if episode> 0:
			summary_str = sess.run(merged_summary, feed_dict={reward_var: r_epi, loss_var:loss, value_var:value})
			summary_writer.add_summary(summary_str, episode)

		if t >500:
                        succ_writer.writerow([episode,count])
                        succ_file.flush()

		# save progress every 500 episodes
		if (episode+1) % 1300  == 0 :
			trainable_saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step = episode)

		if len(loop_time_buf) == 0:
			print("EPISODE", episode, "/ REWARD", r_epi, "/ steps ", T)
		else:
			print("EPISODE", episode, "/ REWARD", r_epi, "/ steps ", T,
				"/ LoopTime:", np.mean(loop_time_buf), "/ loss", loss, "/value",value)

		episode = episode + 1

def main():
	trainNetwork()

if __name__ == "__main__":
        main()
