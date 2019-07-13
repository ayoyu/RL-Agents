import tensorflow as tf 
import numpy as np 


class A2CAgent:

	def __init__(self, obs_shape, n_action, save_steps, gamma=0.99):

		self.obs_shape = obs_shape
		self.n_action = n_action
		self.save_steps = save_steps
		g = tf.Graph()
		with g.as_default():
			with tf.variable_scope('Placeholders'):
				self.states = tf.placeholder(tf.float32, shape=(None,)+obs_shape, name='states')
				self.next_states = tf.placeholder(tf.float32, shape=(None,)+obs_shape, name='next_states')
				self.actions = tf.placeholder(tf.int32, name='actions')
				self.reward = tf.placeholder(tf.float32, name='rewars')
				self.is_done = tf.placeholder(tf.float32, name='is_done')

			with tf.variable_scope('Network'):
				W_hidden = tf.get_variable('W_hidden', shape=obs_shape+(50,), initializer=tf.glorot_uniform_initializer,
					trainable=True)
				b = tf.get_variable('b', shape=[1, 50], trainable=True)

				hidden = tf.nn.relu(tf.add(tf.matmul(self.states, W_hidden), b))
				hidden_next_s = tf.nn.relu(tf.add(tf.matmul(self.next_states, W_hidden), b))

				W_logits = tf.get_variable('W_logits', shape=(hidden.get_shape()[1], n_action), initializer=tf.glorot_uniform_initializer,
					trainable=True)
				b_logits = tf.get_variable('b_logits', shape=[1, n_action], trainable=True)


				W_value = tf.get_variable('W_value', shape=(hidden.get_shape()[1], 1), initializer=tf.glorot_uniform_initializer,
					trainable=True)
				b_value = tf.get_variable('b_value', shape=[1, 1], trainable=True)

				state_value = tf.add(tf.matmul(hidden, W_value), b_value)[:, 0]
				next_state_value = tf.add(tf.matmul(hidden_next_s, W_value), b_value)[:, 0]
				next_state_value = next_state_value * (1 - self.is_done)

				logits = tf.add(tf.matmul(hidden, W_logits), b_logits)
				self.policy = tf.nn.softmax(logits, name='Policy')
				log_policy = tf.nn.log_softmax(logits)
				Advantage = self.reward + gamma * next_state_value - state_value
				J = tf.reduce_mean(tf.reduce_sum(log_policy * tf.one_hot(self.actions, n_action), axis=1) * tf.stop_gradient(Advantage))
				entropy = - tf.reduce_mean(tf.reduce_sum(self.policy * log_policy, axis=1))
				target_value = self.reward + gamma * next_state_value
				loss_critic = tf.reduce_mean((tf.stop_gradient(target_value) - state_value)**2)
				loss = - J + 0.5 * loss_critic - 0.001 * entropy
				weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
				self.optimizer = tf.train.AdamOptimizer().minimize(loss, var_list=weights)
			self.init = tf.global_variables_initializer()
			self.saver = tf.train.Saver()
		self.sess = tf.Session(graph=g)


	def get_action_proba(self, state):
		proba = self.sess.run(self.policy, feed_dict={self.states: [state]})
		return proba[0]


	def train(self, state, reward, next_s, action, done):
		feed_dict = {
		self.states: [state],
		self.next_states : [next_s],
		self.actions: action,
		self.reward: reward,
		self.is_done: done
		}
		self.sess.run(self.optimizer, feed_dict=feed_dict)


	def save_model(self):
		self.saver.save(self.sess, 'a2C_agent', global_step=self.save_steps)


	def load_model(self):
		self.saver.restore(self.sess, f'a2C_agent-{self.save_steps}')


def generate_session(env, agent, t_max=1000):
	s = env.reset()
	sum_rewards = 0
	for _ in range(t_max):
		proba = agent.get_action_proba(s)
		action = np.random.choice(range(agent.n_action), p=proba)
		next_s, r, done, _ = env.step(action)
		sum_rewards += r
		agent.train(s, r, next_s, action, done)
		s = next_s
		if done: break
	return sum_rewards