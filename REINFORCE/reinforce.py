import gym
import tensorflow as tf 
from keras.layers import Dense


class AgentNetwork:

	def __init__(self, state_shape, n_action):

		self.n_action = n_action
		self.state_shape = state_shape

		with tf.scope_variable('Network'):
			self.first_hidden = Dense(units=50, activation='relu', input_shape=state_shape)
			self.logits = Dense(units=n_action, activation='linear')

		with tf.scope_variable('placeholders'):
			self.states = tf.placeholders('float32', shape=(None,)+state_shape, name='states')
			self.cum_rewards  = tf.placeholder('float32', shape=(None,), name='cum_rewards')
			self.actions = tf.placeholder('int32', shape=(None,), name='actions')

		self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Network')


	def get_policy(self, state):

		x = self.first_hidden(state)
		logits = self.logits(x)
		policy = tf.nn.softmax(logits)
		log_policy = tf.nn.log_softmax(logits)
		return policy, log_policy


	def get_action_proba(self, state):
		policy, _ = self.get_policy(state)
		return policy[0]


	def train(self, state):
		policy, log_policy = self.get_policy(state)
		entropy = tf.reduce_mean(tf.reduce_sum(policy * log_policy, axis=-1))
		log_p = tf.reduce_sum(log_policy * tf.one_hot(self.actions, self.n_action), axis=-1)
		J = tf.reduce_mean(log * self.cum_rewards)
		loss = - J - 0.01 * entropy
		tf.train.AdamOptimizer().minimize(loss, var_list=self.weights)
		return loss


def cumulative_rewards(rewards, gamma=0.99):
	"""
	using the recursive formula: G_t = R_t + gamma * G_t+1
	"""
	last_reward = rewards[-1]
	cum_rewards = [last_reward]
	for i in range(len(rewards) -2, -1, -1):
		cum_r = rewards[i] + gamma * cum_rewards[0]
		cum_rewards.insert(0, cum_r)
	return cum_rewards







		


