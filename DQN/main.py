from keras.models import load_model
from DQLearning import AI_Agent
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import gym

env = gym.make("MountainCar-v0")

hidden_units = [18, 10]
agent = AI_Agent(env, 50, 0.1, hidden_units, 0.95, 0.6, 1000, 100)
agent.train()
agent.save_model()
rewards = agent.rewards
episodes = agent.episodes
plt.plot(episodes, rewards)
plt.show()



