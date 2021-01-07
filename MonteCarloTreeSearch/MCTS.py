from math import log, sqrt
import gym
from SnapshotENV import SnapshotEnv
import pickle
import numpy as np
from itertools import count

env = SnapshotEnv(gym.make('CartPole-v0').env)
n_actions = env.action_space.n
SCALE_UCB = 10
MAX_VALUE = 1e100

class Node:
	parent = None
	value_sum = 0
	times_visited = 0

	def __init__(self, parent, action):
		self.parent = parent
		self.action = action
		self.children = set()
		res = env.get_result(self.parent.snapshot, action)
		self.snapshot, self.obs, self.immediate_reward, self.is_done, _ = res
		

	def __repr__(self):
		return f'Node({self.parent}, {self.action})'


	def is_root(self):
		return self.parent is None


	def is_leaf(self):
		return len(self.children) == 0


	def get_mean_value(self):
		return self.value_sum / self.times_visited if self.times_visited != 0 else 0.


	def ucb_score(self):
		U = 2 * sqrt(log(self.parent.times_visited) / self.times_visited) if self.times_visited != 0 else MAX_VALUE
		return self.get_mean_value() + SCALE_UCB * U


	def selection(self):
		if self.is_leaf():
			return self
		children = list(self.children)
		best_leaf = children[np.argmax([child.ucb_score() for child in children])]
		return best_leaf.selection()


	def expand(self):
		if self.times_visited == 0:
			return self
		for action in range(n_actions):
			self.children.add(Node(self, action))
		assert not self.is_done, "the episode is finished! can't expand"

		return self.selection()


	def rollout(self, t_max=10**4):
		if self.is_done:
			return 0.
		env.load_snapshot(self.snapshot)
		total_reward_rollout = 0
		for _ in range(t_max):
			action = env.action_space.sample()
			next_s, r, done, _ = env.step(action)
			total_reward_rollout += r
			if done: break
		return total_reward_rollout


	def back_propagate(self, rollout_reward):
		node_value = self.immediate_reward + rollout_reward
		self.value_sum += node_value
		self.times_visited += 1

		if not self.is_root():
			self.parent.back_propagate(rollout_reward)


	def safe_delete(self):
		"""
		for deleting unnecessary node
		"""
		del self.parent
		for child in self.children:
			child.safe_delete()
			del child


class Root(Node):
	"""
	creates special node that acts like tree root
    snapshot: snapshot (from env.get_snapshot) to start planning from
    observation: last environment observation
	"""
	def __init__(self, snapshot, obs):

		self.parent = self.action = None
		self.snapshot = snapshot
		self.obs = obs
		self.children = set()
		self.ucb_score = 0
		self.immediate_reward = 0
		self.is_done = False


	@staticmethod
	def to_root(node):
		"""initializes node as root"""
		root = Root(node.snapshot, node.obs)
		attr_names = ["value_sum", "times_visited",
		"children", "is_done"]
		for attr in attr_names:
			setattr(root, attr, getattr(node, attr))
		return root


def plan_mcts(root, n_iter=1000):
	"""
	builds tree with mcts for n_iter
	root: tree root to plan from
	n_iter: how many select->expand->rollout->back_propagate
	"""
	for _ in range(n_iter):
		node = root.selection()

		if node.is_done:
			node.back_propagate(0)
		else:
			best_leaf = node.expand()
			rollout_reward = best_leaf.rollout()
			best_leaf.back_propagate(rollout_reward)


if __name__ == '__main__':
	env = SnapshotEnv(gym.make("CartPole-v0").env)
	root_obs = env.reset()
	root_snapshot = env.get_snapshot()
	root = Root(root_snapshot, root_obs)
	plan_mcts(root, n_iter=1000)
	test_env = pickle.loads(root_snapshot) # env used to show progress
	total_reward = 0
	while True:

		children = list(root.children)
		best_child = children[np.argmax([child.get_mean_value() for child in children])]

		s, r, done, _ = test_env.step(best_child.action)
		print(f"current reward: {r} | total_reward: {total_reward}")
		test_env.render()
		assert (best_child.obs == s).all()
		total_reward += r
		if done:
			print(f"finished with reward: {total_reward}")
			test_env.close()
			break
		# no need for the other part of the tree because the role of
		# root will be transmited to the best_child
		for child in children:
			if child != best_child:
				child.safe_delete()

		root = Root.to_root(best_child)
		if root.is_leaf():
			plan_mcts(root, n_iter=3000)

		best_leaf = root.expand()
		child_value = root.rollout()
		root.back_propagate(child_value)