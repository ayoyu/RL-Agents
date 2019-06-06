from keras.models import load_model
from DQLearning import AI_Agent
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import gym

env = gym.make("MountainCar-v0")

units = [50, 30]
agent = AI_Agent(env, 5000, 0.1, units, 0.95, 0.2, 50)
agent.train()
agent.save_model()
rewards = agent.rewards
episodes = agent.episodes
plt.plot(episodes, rewards)
plt.show()



