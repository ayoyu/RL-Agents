# RL Agents

- Qlearning (table version)
- Deep Qlearning (Q_network & Q_target)
- Monte Carlot Tree Search 
- REINFORCE (Policy gradient)
- Advantage actor critic (A2C)

## Gym Environements:

### Mountain Car Problem: [OpenAI-Gym](https://gym.openai.com/envs/MountainCar-v0/)

![MountainCar](MountainCar.jpeg)

A car is on a one-dimensional track, positioned between two "mountains". The goal is to drive up the mountain on the right; however, the car's engine is not strong enough to scale the mountain in a single pass. Therefore, the only way to succeed is to drive back and forth to build up momentum.

The goal then is to train an agent with reinforcement learning to solve this task.

### CartPol Problem: [OpenAI-Gym](https://gym.openai.com/envs/CartPole-v0/)

![Cartpol](cartpol.gif)

A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright. The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.

### Breakout-v0: [OpenAI-Gym](https://gym.openai.com/envs/Breakout-v0/)
![Breakout-v0](breakout-v0.gif)

Maximize your score in the Atari 2600 game Breakout. In this environment, the observation is an RGB image of the screen, which is an array of shape (210, 160, 3) Each action is repeatedly performed for a duration of kk frames, where kk is uniformly sampled from \{2, 3, 4\}{2,3,4}.

## Create the virtual env (recommended)::

```
$ virtualenv -p python3.6 myenv
$ source myenv/bin/activate
(myenv)$ pip install -r requirements.txt
```

### Qlearnig (MountainCar-v0):

#### Train and save the agent:

```
$ python3 ./QLearning/run.py
```

#### load the agent and play:

```
$ python3 ./QLearning/Evaluate.py
```

### DQN (Breakout-v0): (see the ipynb file)

### Monte Carlo Tree Search (Cartpol):
```
$ python3 ./MonteCarloTreeSearch/MCTS.py
```

### REINFORCE (CartPole):
```
$ python3 ./REINFORCE/main.py
```

### A2C (CartPole):
#### Train and save the agent:
```
$ python3 ./Advantage_Actor_Critic/run.py
```
#### load the agent and play:

```
$ python3 ./Advantage_Actor_Critic/Evaluate.py
```