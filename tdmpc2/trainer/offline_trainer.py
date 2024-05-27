import os
from copy import deepcopy
from time import time
from pathlib import Path
from glob import glob

import numpy as np
import torch
from tqdm import tqdm

from common.buffer import Buffer, RobomimicBuffer
from trainer.base import Trainer


class OfflineTrainer(Trainer):
	"""Trainer class for multi-task offline TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._start_time = time()
	
	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		results = dict()
		for task_idx in tqdm(range(len(self.cfg.tasks)), desc='Evaluating'):
			ep_rewards, ep_successes = [], []
			for _ in range(self.cfg.eval_episodes):
				obs, done, ep_reward, t = self.env.reset(task_idx), False, 0, 0
				while not done:
					action = self.agent.act(obs, t0=t==0, eval_mode=True, task=task_idx)
					obs, reward, done, info = self.env.step(action)
					ep_reward += reward
					t += 1
				ep_rewards.append(ep_reward)
				ep_successes.append(info['success'])
			results.update({
				f'episode_reward+{self.cfg.tasks[task_idx]}': np.nanmean(ep_rewards),
				f'episode_success+{self.cfg.tasks[task_idx]}': np.nanmean(ep_successes),})
		return results

	def load_data(self):
		# Load data
		assert self.cfg.task in self.cfg.data_dir, \
			f'Expected data directory {self.cfg.data_dir} to contain {self.cfg.task}, ' \
			f'please double-check your config.'

		# Create buffer for sampling
		_cfg = deepcopy(self.cfg)
		if _cfg.task.startswith("mt"):
			_cfg.episode_length = 101 if self.cfg.task == 'mt80' else 501
			_cfg.buffer_size = 550_450_000 if self.cfg.task == 'mt80' else 345_690_000
			self.buffer = Buffer(_cfg)
			fp = Path(os.path.join(self.cfg.data_dir, '*.pt'))
		elif _cfg.task.startswith("myo"):
			_cfg.episode_length = 101
			_cfg.buffer_size = 10_000_000
			self.buffer = RobomimicBuffer(_cfg)
			fp = Path(os.path.join(self.cfg.data_dir, '*.hdf5'))
		_cfg.steps = _cfg.buffer_size
		
		fps = sorted(glob(str(fp)))
		assert len(fps) > 0, f'No data found at {fp}'
		print(f'Found {len(fps)} files in {fp}')
	
		for fp in tqdm(fps, desc='Loading data'):
			if _cfg.task.startswith("mt"):
				td = torch.load(fp)
			else:
				td = self.buffer.load_hdf5(fp)
			assert td.shape[1] == _cfg.episode_length, \
				f'Expected episode length {td.shape[1]} to match config episode length {_cfg.episode_length}, ' \
				f'please double-check your config.'
			for i in range(len(td)):
				self.buffer.add(td[i])
		assert self.buffer.num_eps == self.buffer.capacity, \
			f'Buffer has {self.buffer.num_eps} episodes, expected {self.buffer.capacity} episodes.'
				
	def train(self):
		"""Train a TD-MPC2 agent."""
		assert self.cfg.multitask and self.cfg.task in {'mt30', 'mt80', 'myo10'}, \
			'Offline training only supports multitask training with mt30, mt80, or myo10 task sets.'

		self.load_data()
		
		print(f'Training agent for {self.cfg.steps} iterations...')
		metrics = {}
		for i in range(self.cfg.steps):

			# Update agent
			train_metrics = self.agent.update(self.buffer)

			# Evaluate agent periodically
			if i % self.cfg.eval_freq == 0 or i % 10_000 == 0:
				metrics = {
					'iteration': i,
					'total_time': time() - self._start_time,
				}
				metrics.update(train_metrics)
				if i % self.cfg.eval_freq == 0:
					metrics.update(self.eval())
					self.logger.pprint_multitask(metrics, self.cfg)
					if i > 0:
						self.logger.save_agent(self.agent, identifier=f'{i}')
				self.logger.log(metrics, 'pretrain')
			
		self.logger.finish(self.agent)
