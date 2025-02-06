import gymnasium as gym
import gym_sorting
import gym_stacking
import numpy as np
from gymnasium.spaces import Dict, Box

class D3ILWrapper(gym.Wrapper):
	def __init__(self, env, cfg):
		super().__init__(env)
		self.env = env
		self.task = cfg.task
		self.cfg = cfg
		self.max_episode_steps = env.max_steps_per_episode
		state_shape = env.observation_space["agent_pos"].shape
		if 'd3il-sorting' in cfg.task:
			self.observation_space = Dict({
				"state": Box(low=-np.inf, high=np.inf, shape=state_shape)
			})
		else:
			env_state_shape = env.observation_space["environment_state"].shape
			concatenated_shape = (state_shape[0] + env_state_shape[0],)
			self.observation_space = Dict({
				"state": Box(low=-np.inf, high=np.inf, shape=concatenated_shape)
			})

	def reset(self):
		obs = self.env.reset()[0]
		agent_pos = obs["agent_pos"]
		if 'd3il-sorting' in self.task:
			return agent_pos
		else:
			environment_state = obs["environment_state"]
			return np.concatenate([agent_pos, environment_state])

	def step(self, action):
		obs, reward, terminated, truncated, info = self.env.step(action.copy())
		agent_pos = obs["agent_pos"]
		if 'd3il-sorting' in self.task:
			state = agent_pos
		else:
			environment_state = obs["environment_state"]
			state = np.concatenate([agent_pos, environment_state])
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
	if "d3il-sorting" in cfg.task:
		return D3ILWrapper(
			gym.make("gym_sorting/sorting-v0", disable_env_checker=True, **kwargs),
			cfg)
	elif cfg.task == "d3il-stacking":
		return D3ILWrapper(
			gym.make("gym_stacking/stacking-v0", disable_env_checker=True, **kwargs),
			cfg)
	else:
		raise ValueError('Unknown task:', cfg.task)
