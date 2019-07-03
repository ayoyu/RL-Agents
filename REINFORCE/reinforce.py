import gym
import tensorflow as tf 
from keras.layers import Dense
import numpy as np

class AgentNetwork:

	def __init__(self, state_shape, n_action):

		self.n_action = n_action
		self.state_shape = state_shape
		g = tf.Graph()

		with g.as_default():
			tf.set_random_seed(10)
			with tf.variable_scope('placeholders'):
				
				self.states = tf.placeholder('float32', shape=(None,)+state_shape, name='states')
				self.cum_rewards  = tf.placeholder('float32', shape=(None,), name='cum_rewards')
				self.actions = tf.placeholder('int32', shape=(None,), name='actions')

			with tf.variable_scope('Network'):

				first_hidden = Dense(units=50, activation='relu')(self.states)
				logits = Dense(units=n_action, activation='linear')(first_hidden)
				self.policy = tf.nn.softmax(logits)
				log_policy = tf.nn.log_softmax(logits)
				entropy = tf.reduce_mean(tf.reduce_sum(self.policy * log_policy, axis=-1))
				log_p = tf.reduce_sum(log_policy * tf.one_hot(self.actions, self.n_action), axis=-1)
				J = tf.reduce_mean(log_p * self.cum_rewards)
				loss = - J - 0.01 * entropy
				weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Network')
				self.optimizer = tf.train.AdamOptimizer().minimize(loss, var_list=weights)

			self.init = tf.global_variables_initializer()

		self.sess = tf.Session(graph=g)


	def get_action_proba(self, state):
		state = np.reshape(state, (1, state.shape[0]))
		proba = self.sess.run(self.policy, feed_dict={self.states: state})
		return proba[0]


	@staticmethod
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


	def train(self, actions, states, rewards):
		cum_rewards = AgentNetwork.cumulative_rewards(rewards)
		self.sess.run(self.optimizer, feed_dict={
			self.states: states, self.actions: actions, self.cum_rewards: cum_rewards
			})



def generate_session(agent, env, t_max=1000):
	states, actions, rewards = [], [], []
	s = env.reset()
	for _ in range(t_max):
		agent_prob = agent.get_action_proba(s)
		action = np.random.choice(range(agent.n_action), p=agent_prob)
		next_s, r, done, _ = env.step(action)
		states.append(s)
		actions.append(action)
		rewards.append(r)
		s = next_s
		if done: break
	
	agent.train(actions, states, rewards)
	return sum(rewards)