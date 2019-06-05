import gym
import pickle
import time
import os
from Q_learning import Q_learning

current_dir = os.path.dirname(os.path.realpath(__file__))
env = gym.make('MountainCar-v0')

with open(os.path.join(current_dir, 'Q_table.pkl'), 'rb') as file:
	Model = pickle.load(file)

state = env.reset()
done = False
step = 0
while not done:
	step += 1
	action = Model.make_decision(state)
	new_state, reward, done, _ = env.step(action)
	print(reward, new_state)
	state = new_state
	env.render()
if state[0] >= env.goal_position:
	print(f'we made it at the step {step}')
time.sleep(1)
env.close()