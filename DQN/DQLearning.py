import random
import os
import numpy as np
import gym
import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

SHOW_EVERY = 500

class DNN:

	def __init__(self, learning_rate, hidden_units, nbr_feature, output_shape):

		self.learning_rate = learning_rate
		self.hidden_units = hidden_units
		self.nbr_feature = nbr_feature
		self.output_shape = output_shape
		self.model_structure = self.build()


	def build(self):
		model = Sequential()
		model.add(Dense(units=50, input_shape=(self.nbr_feature,),
			use_bias=True, kernel_initializer='glorot_uniform', activation='relu'))
		for units in self.hidden_units:
			model.add(Dense(units=units, use_bias=True,
				kernel_initializer='glorot_uniform', activation='relu'))
		model.add(Dense(units=self.output_shape, activation='linear', use_bias=True, kernel_initializer='glorot_uniform'))		
		model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
		return model


class DQN:

	def __init__(self, dim_obs_space, dim_action_space, learning_rate,
		hidden_units, gamma, batch):

		self.dim_obs_space = dim_obs_space
		self.dim_action_space = dim_action_space
		self.learning_rate = learning_rate
		self.hidden_units = hidden_units
		self.gamma = gamma
		self.batch = batch
		Network = DNN(learning_rate, hidden_units, dim_obs_space, dim_action_space)
		self.model = Network.model_structure
		self.memory = list()


	def add_to_memory(self, state, action, next_state, reward, done):
		self.memory.append((state, action, next_state, reward, done))


	def take_action(self, state, epsilon):

		if np.random.random() <= epsilon:
			action = np.random.randint(0, self.dim_action_space)
		else:
			action = np.argmax(self.model.predict(state)[0])
		return action


	def fit_policy(self):
		if len(self.memory) < self.batch:
			return
		else:
			mini_batch_data = random.sample(self.memory, self.batch)
			for args in mini_batch_data:
				state, action, next_state, reward, done = args
				max_future_reward = reward
				if not done:
					max_future_reward = reward + self.gamma * np.max(self.model.predict(next_state)[0])
				
				Q_value = self.model.predict(state)[0]
				Q_value[action] = max_future_reward
				Q_value = np.reshape(Q_value, (1, Q_value.shape[0]))
				self.model.fit(state, Q_value)


class AI_Agent:

	def __init__(self, env, episodes, learning_rate, hidden_units,
		gamma, epsilon, bacth):

		self.env = env
		self.episodes = episodes
		self.learning_rate = learning_rate
		self.hidden_units = hidden_units
		self.epsilon = epsilon
		self.gamma = gamma
		self.bacth = bacth
		dim_obs_space = env.observation_space.shape[0]
		dim_action_space = env.action_space.n
		self.DQN_agent = DQN(dim_obs_space, dim_action_space, learning_rate, hidden_units,
			gamma, bacth)
		self.rewards = list()


	def train(self):
		start_epsilon_decay = 1
		end_epsilon_decay = self.episodes // 4
		decay_value = self.epsilon / (end_epsilon_decay - start_epsilon_decay)

		for episode in range(0, self.episodes):
			state = self.env.reset()
			state = np.reshape(state, (1, state.shape[0]))
			done = False
			reward_ep = 0
			while not done:
				action = self.DQN_agent.take_action(state, self.epsilon)
				next_state, reward, done, _ = self.env.step(action)
				reward_ep += reward
				next_state = np.reshape(next_state, (1, next_state.shape[0]))
				self.DQN_agent.add_to_memory(state, action, next_state, reward, done)
				state = next_state
				if episode % SHOW_EVERY == 0:
					self.env.render()
				if not done:
					self.DQN_agent.fit_policy()
				elif state[0][0] >= self.env.goal_position:
					print(f"we made it at episode {episode}")
			self.rewards.append(reward_ep)
			self.epsilon -= decay_value
		self.env.close()


	def get_model(self):
		return self.DQN_agent.model


	def save_model(self):
		current_dir = os.path.dirname(os.path.realpath(__file__))
		model_dir = os.path.join(current_dir, 'Models')
		if not os.path.exists(model_dir):
			os.mkdir(model_dir)
		self.DQN_agent.model.save(os.path.join(model_dir, "DQN_model.h5"))
