import numpy as np 
from A2C import A2CAgent, generate_session
import gym
import matplotlib.pyplot as plt


env = gym.make("CartPole-v0").env
n_action = env.action_space.n 
state_shape = env.observation_space.shape
steps = 100
agent = A2CAgent(state_shape, n_action, steps)

agent.sess.run(agent.init)
Rewards = []
for _ in range(100):
	sum_reward = 0
	for _ in range(steps):
		r = generate_session(env, agent)
		sum_reward += r
	mean_reward = sum_reward / steps
	Rewards.append(mean_reward)
	agent.save_model()
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