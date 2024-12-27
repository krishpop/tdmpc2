import gymnasium as gym
import gym_sorting
import numpy as np
from gymnasium.spaces import Dict, Box

class D3ILWrapper(gym.Wrapper):
	def __init__(self, env, cfg):
		super().__init__(env)
		self.env = env
		self.cfg = cfg
		self.max_episode_steps = env.max_steps_per_episode
		obs_shape = env.observation_space["agent_pos"].shape
		self.observation_space = Dict({
			"state": Box(low=-np.inf, high=np.inf, shape=obs_shape)
		})

	def reset(self):
		state = self.env.reset()[0]["agent_pos"]
		return state

	def step(self, action):
		obs, reward, terminated, truncated, info = self.env.step(action.copy())
		state = obs["agent_pos"]
		info['success'] = info['is_success']
		return state, reward, terminated or truncated, info

	@property
	def unwrapped(self):
		return self.env.unwrapped

	def render(self, *args, **kwargs):
		return self.env.render()


def make_env(cfg):
	"""
	Make D3IL environment.
	"""
	kwargs = {
		"max_steps_per_episode": 1000,
		"if_vision": False,
		"render": False,
		"self_start": True
	}
	if cfg.task == "d3il-sorting":
		return D3ILWrapper(
			gym.make("gym_sorting/sorting-v0", disable_env_checker=True, **kwargs),
			cfg)
	else:
		raise ValueError('Unknown task:', cfg.task)
