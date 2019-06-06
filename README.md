# Reinforcement_learning_apps

#### Mountain Car Problem:

![MountainCar](MountainCar.jpeg)

A car is on a one-dimensional track, positioned between two "mountains". The goal is to drive up the mountain on the right; however, the car's engine is not strong enough to scale the mountain in a single pass. Therefore, the only way to succeed is to drive back and forth to build up momentum.[OpenAI](https://gym.openai.com/envs/MountainCar-v0/)

The goal then is to train an agent with reinforcement learning to solve this task.

##### install dependencies:

```
$ pip install -r requirements.txt
```

#### Qlearnig:

##### Train and save the agent:

```
$ cd ./QLearning
$ python3 Q_learning.py
```

##### load the agent and make decision for a given state:

```
$ python3 test_Qtable.py
```

#### Deep Qlearning:

##### Train and save the agent:

```
$ cd ./DQN
$ python3 main.py
```