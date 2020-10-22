import os
import numpy as np
import tensorflow as tf
from eval.evaluate import accuracy
from tensorflow.contrib import slim
from loss.loss import cross_entropy_loss
 
class TextCNN(object):
	def __init__(self,
				 num_classes,
				 seq_length,
				 vocab_size,
				 embedding_dim,
				 learning_rate,
				 learning_decay_rate,
				 learning_decay_steps,
				 epoch,
				 filter_sizes,
				 num_filters,
				 dropout_keep_prob,
				 l2_lambda
				 ):
		self.num_classes = num_classes
		self.seq_length = seq_length
		self.vocab_size = vocab_size
		self.embedding_dim = embedding_dim
		self.learning_rate = learning_rate
		self.learning_decay_rate = learning_decay_rate
		self.learning_decay_steps = learning_decay_steps
		self.epoch = epoch
		self.filter_sizes = filter_sizes
		self.num_filters = num_filters
		self.dropout_keep_prob = dropout_keep_prob
		self.l2_lambda = l2_lambda
		self.input_x = tf.placeholder(tf.int32, [None, self.seq_length], name='input_x')
		self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name='input_y')
		self.l2_loss = tf.constant(0.0)
		self.model()

	def model(self):
		#embedding 
		with tf.name_scope("embedding"):
			self.embedding= tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_dim], -1.0, 1.0),
										name="embedding")
			self.embedding_inputs = tf.nn.embedding_lookup(self.embedding,self.input_x)
			self.embedding_inputs = tf.expand_dims(self.embedding_inputs,-1)
 
				 # convolution layer + pooling layer
		pooled_outputs = []
		for i, filter_size in enumerate(self.filter_sizes):
			with tf.name_scope("conv_{0}".format(filter_size)):
				filter_shape = [filter_size, self.embedding_dim, 1, self.num_filters]
				W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
				b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
				conv = tf.nn.conv2d(
					self.embedding_inputs,
					W,
					strides=[1, 1, 1, 1],
					padding="VALID",
					name="conv"
				)
				h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
				pooled = tf.nn.max_pool(
					h,
					ksize=[1, self.seq_length - filter_size + 1, 1, 1],
					strides=[1, 1, 1, 1],
					padding='VALID',
					name="pool"
				)
				pooled_outputs.append(pooled)
 
				 # splicing the feature vectors obtained from the convolution kernel of each size
		num_filters_total = self.num_filters * len(self.filter_sizes)
		h_pool = tf.concat(pooled_outputs, 3)
		h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
 
				 # Dropout the resulting sentence vector
		with tf.name_scope("dropout"):
			h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)
 
				 # 
		with tf.name_scope("output"):
			W = tf.get_variable("W",shape=[num_filters_total, self.num_classes],
								initializer=tf.contrib.layers.xavier_initializer())
			b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")
			self.l2_loss += tf.nn.l2_loss(W)
			self.l2_loss += tf.nn.l2_loss(b)
			self.logits = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
			self.pred = tf.argmax(self.logits, 1, name="predictions")
 
				 #loss function
		self.loss = cross_entropy_loss(logits=self.logits, labels=self.input_y) + self.l2_lambda*self.l2_loss
 
				 # optimization function
		self.global_step = tf.train.get_or_create_global_step()
		learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,
												   self.learning_decay_steps, self.learning_decay_rate,
												   staircase=True)
 
		optimizer = tf.train.AdamOptimizer(learning_rate)
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		self.optim = slim.learning.create_train_op(total_loss=self.loss, optimizer=optimizer, update_ops=update_ops)
 
				 # Accuracy 
		self.acc = accuracy(logits=self.logits, labels=self.input_y)
 
	def fit(self,train_x,train_y,val_x,val_y,batch_size):
		#Create model save path
		if not os.path.exists('./saves/textcnn'): os.makedirs('./saves/textcnn')
		if not os.path.exists('./train_logs/textcnn'): os.makedirs('./train_logs/textcnn')

		train_steps = 0
		best_val_acc = 0
		# summary
		tf.summary.scalar('val_loss', self.loss)
		tf.summary.scalar('val_acc', self.acc)
		merged = tf.summary.merge_all()
 
		# Initialize variables 
		sess = tf.Session()
		writer = tf.summary.FileWriter('./train_logs/textcnn', sess.graph)
		saver = tf.train.Saver(max_to_keep=10)
		sess.run(tf.global_variables_initializer())
 
		for i in range(self.epoch):
			batch_train = self.batch_iter(train_x, train_y, batch_size)
			for batch_x,batch_y in batch_train:
				train_steps += 1
				feed_dict = {self.input_x:batch_x,self.input_y:batch_y}
				_, train_loss, train_acc = sess.run([self.optim,self.loss,self.acc],feed_dict=feed_dict)
 
				if train_steps % 1000 == 0:
					feed_dict = {self.input_x:val_x,self.input_y:val_y}
					val_loss,val_acc = sess.run([self.loss,self.acc],feed_dict=feed_dict)
 
					summary = sess.run(merged,feed_dict=feed_dict)
					writer.add_summary(summary, global_step=train_steps)
 
					if val_acc>=best_val_acc:
						best_val_acc = val_acc
						saver.save(sess, "./saves/textcnn/", global_step=train_steps)
 
					msg = 'epoch:%d/%d,train_steps:%d,train_loss:%.4f,train_acc:%.4f,val_loss:%.4f,val_acc:%.4f'
					print(msg % (i,self.epoch,train_steps,train_loss,train_acc,val_loss,val_acc))
 
		sess.close()
 
	def batch_iter(self, x, y, batch_size=32, shuffle=True):
		"""
		Generate batch data
		:param x: training set feature variable
		:param y: training set label
		:param batch_size: the size of each batch
		:param shuffle: Whether to disrupt data at every epoch
		:return:
		"""
		data_len = len(x)
		num_batch = int((data_len - 1) / batch_size) + 1
 
		if shuffle:
			shuffle_indices = np.random.permutation(np.arange(data_len))
			x_shuffle = x[shuffle_indices]
			y_shuffle = y[shuffle_indices]
		else:
			x_shuffle = x
			y_shuffle = y
		for i in range(num_batch):
			start_index = i * batch_size
			end_index = min((i + 1) * batch_size, data_len)
			yield (x_shuffle[start_index:end_index], y_shuffle[start_index:end_index])
 
	def predict(self,x):
		sess = tf.Session()
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver(tf.global_variables())
		ckpt = tf.train.get_checkpoint_state('./saves/textcnn/')
		saver.restore(sess, ckpt.model_checkpoint_path)
 
		feed_dict = {self.input_x: x}
		logits = sess.run(self.logits, feed_dict=feed_dict)
		y_pred = np.argmax(logits, 1)
		return y_pred
