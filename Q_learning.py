import gym
import numpy as np
import pickle
import os


env = gym.make('MountainCar-v0')
# print(env.action_space)
# print(env.observation_space)
# print(env.observation_space.low)
# print(env.observation_space.high)
# print(env._max_episode_steps)
# print(env.goal_position)
# print(env.action_space.n)
# print(dir(env))
"""
Discrete(3)
Box(2,)
[-1.2  -0.07]
[0.6  0.07]
200
0.5

"""

class Q_learning:
	def __init__(self, learning_rate, discount_rate, episodes, epsilon, discretize_count, show):
		
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
		new_state = (state - env.observation_space.low) / self.lenght_interval
		return tuple(new_state.astype(np.int))


	def train(self):
		start_decay_epsilon = 1
		end_decay_epsilon = self.episodes // 4
		decay_value = self.epsilon / (end_decay_epsilon - start_decay_epsilon)
		for episode in range(0, self.episodes):
			state = env.reset()
			discret_state = self.get_discretize_state(state)
			done = False
			while not done:
				if np.random.random() < self.epsilon:
					action = np.random.randint(0, env.action_space.n)
				else:
					action = np.argmax(self.q_table[discret_state])

				new_state, reward, done, _ = env.step(action)
				discret_new_state = self.get_discretize_state(new_state)
				if episode % self.show == 0:
					render = True
				else:
					render = False
				if render:
					env.render()
				if not done:
					max_future_reward = np.max(self.q_table[discret_new_state])
					current_q = self.q_table[discret_state + (action, )]
					q_value = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount_rate * max_future_reward)
					self.q_table[discret_state + (action,)] = q_value
				elif new_state[0] >= env.goal_position:
					print(f"Made it at {episode} episode")

				discret_state = discret_new_state
			self.epsilon -= decay_value
		env.close()
		return self


	def make_decision(self, given_state):
		discret_state = self.get_discretize_state(given_state)
		action = np.argmax(self.q_table[discret_state])
		return action


if __name__ == '__main__':
	current_dir = os.path.dirname(os.path.realpath(__file__))
	Model = Q_learning(0.1, 0.95, 7000, 0.1, 10, 500)
	Model.train()
	with open(os.path.join(current_dir, 'Q_table.pkl'), 'wb') as file:
		pickle.dump(Model, file)




			















