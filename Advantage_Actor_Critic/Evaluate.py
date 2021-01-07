import os
import numpy as np 
from A2C import AdvantageActorCritic
import gym
import tensorflow as tf
from tqdm import tqdm 


env = gym.make("CartPole-v0").env
n_action = env.action_space.n 
state_shape = env.observation_space.shape
steps = 100
agent = AdvantageActorCritic(state_shape, n_action, steps)
load_path = os.path.join(os.path.dirname(__file__))
agent.load_model(load_path)


def play_game(env, agent, t_max=1000):
	s = env.reset()
	sum_rewards = 0
	for _ in tqdm(range(t_max)):
		proba = agent.get_action_proba(s)
		action = np.random.choice(range(agent.n_action), p=proba)
		next_s, r, done, _ = env.step(action)
		env.render()
		sum_rewards += r
		s = next_s
		if done: break
	env.close()
	return sum_rewards


def evaluate(env, agent):
	steps = agent.save_steps
	sum_reward = 0
	for _ in tqdm(range(steps)):
		r = play_game(env, agent)
		sum_reward += r
	return sum_reward / steps


if __name__ == '__main__':
	mean_rewards = evaluate(env, agent)
	print(f'Mean rewards for {steps} episodes: {mean_rewards}')