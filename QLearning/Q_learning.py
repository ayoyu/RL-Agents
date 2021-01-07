import gym
import numpy as np
import pickle
from tqdm import tqdm


class QAgent:
	def __init__(self, env, learning_rate, discount_rate, episodes,
		epsilon, discretize_count, show):
		
		self.env = env
		self.learning_rate = learning_rate
		self.discount_rate = discount_rate
		self.episodes = episodes
		self.epsilon = epsilon
		self.discretize_count = discretize_count
		self.show = show
		discretize_space = np.array([discretize_count] * len(env.observation_space.low))
		self.lenght_interval = (env.observation_space.high - env.observation_space.low) / discretize_space
		q_table_shape = (*discretize_space, env.action_space.n)
		self.q_table = np.random.uniform(low=-2, high=0, size=q_table_shape)


	def get_discretize_state(self, state):
		new_state = (state - self.env.observation_space.low) / self.lenght_interval
		return tuple(new_state.astype(np.int))


	def train(self):
		start_decay_epsilon = 1
		end_decay_epsilon = self.episodes // 4
		decay_value = self.epsilon / (end_decay_epsilon - start_decay_epsilon)
		for episode in tqdm(range(0, self.episodes)):
			state = self.env.reset()
			discret_state = self.get_discretize_state(state)
			done = False
			while not done:
				if np.random.random() < self.epsilon:
					action = np.random.randint(0, self.env.action_space.n)
				else:
					action = np.argmax(self.q_table[discret_state])

				new_state, reward, done, _ = self.env.step(action)
				discret_new_state = self.get_discretize_state(new_state)
				if episode % self.show == 0:
					render = True
				else:
					render = False
				if render:
					self.env.render()
				if not done:
					max_future_reward = np.max(self.q_table[discret_new_state])
					current_q = self.q_table[discret_state + (action, )]
					q_value = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount_rate * max_future_reward)
					self.q_table[discret_state + (action,)] = q_value
				# elif new_state[0] >= self.env.goal_position:
				# 	print(f"Made it at {episode} episode")

				discret_state = discret_new_state
			self.epsilon -= decay_value
		self.env.close()
		return self


	def make_decision(self, given_state):
		discret_state = self.get_discretize_state(given_state)
		action = np.argmax(self.q_table[discret_state])
		return action