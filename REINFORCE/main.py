import numpy as np
import gym
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 
from reinforce import AgentNetwork, generate_session


env = gym.make("CartPole-v0").env
n_action = env.action_space.n 
state_shape = env.observation_space.shape
agent = AgentNetwork(state_shape, n_action)

agent.sess.run(agent.init)

Rewards = []
for _ in range(100):
	sum_reward = 0
	steps = 100
	for _ in range(steps):
		r = generate_session(agent, env)
		sum_reward += r
	mean_reward = sum_reward / steps
	Rewards.append(mean_reward)

	print(f'mean rewards : {mean_reward}')
	if mean_reward > 300:
		print('You Win')
		plt.title('mean Game rewards')
		plt.xlabel('Epoch Game')
		plt.ylabel('mean reward')
		plt.plot(Rewards)
		plt.show()
		plt.close()
		break