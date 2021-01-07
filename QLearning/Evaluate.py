import gym
import pickle
import time
import os


if __name__ == '__main__':

	current_dir = os.path.dirname(os.path.realpath(__file__))
	env = gym.make('MountainCar-v0')

	with open(os.path.join(current_dir, 'Q_table.pkl'), 'rb') as file:
		agent = pickle.load(file)

	state = env.reset()
	done = False
	step = 0
	while not done:
		step += 1
		action = agent.make_decision(state)
		new_state, reward, done, _ = env.step(action)
		print(reward, new_state)
		state = new_state
		env.render()
	if state[0] >= env.goal_position:
		print(f'We made it at: {step} step')
	time.sleep(1)
	env.close()