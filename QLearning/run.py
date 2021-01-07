from Q_learning import QAgent
import gym
import os
import pickle


if __name__ == '__main__':
	env = gym.make('MountainCar-v0')
	current_dir = os.path.dirname(os.path.realpath(__file__))
	model = QAgent(env, learning_rate=0.1, discount_rate=0.95,
		episodes=7000, epsilon=0.1, discretize_count=10, show=500)
	model.train()
	with open(os.path.join(current_dir, 'Q_table.pkl'), 'wb') as file:
		pickle.dump(model, file)