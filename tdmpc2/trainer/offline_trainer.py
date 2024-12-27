import os
from copy import deepcopy
from time import time
from pathlib import Path
from glob import glob

import numpy as np
import torch
from tqdm import tqdm

from common.buffer import Buffer, RobomimicBuffer, D3ILBuffer
from trainer.base import Trainer
from common.__init__ import TASK_SET

class OfflineTrainer(Trainer):
	"""Trainer class for multi-task offline TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._start_time = time()
	
	def eval(self, step):
		"""Evaluate a TD-MPC2 agent."""
		results = dict()
		ep_rewards, ep_successes = [], []
		for i in range(self.cfg.eval_episodes):
			obs, done, ep_reward, t = self.env.reset(), False, 0, 0
			if self.cfg.save_video:
				self.logger.video.init(self.env, enabled=(i==0))
			while not done:
				torch.compiler.cudagraph_mark_step_begin()
				action = self.agent.act(obs, t0=t==0, eval_mode=True)
				obs, reward, done, info = self.env.step(action)
				ep_reward += reward
				t += 1
				if self.cfg.save_video:
					self.logger.video.record(self.env)
			ep_rewards.append(ep_reward)
			ep_successes.append(info['success'])
			if self.cfg.save_video:
				self.logger.video.save(step)
		results.update({
			f'episode_reward': np.nanmean(ep_rewards),
			f'episode_success': np.nanmean(ep_successes),})
		return results
	
	def _load_dataset(self):
		"""Load dataset for offline training."""
		fp = Path(os.path.join(self.cfg.data_dir, '*.pt'))
		fps = sorted(glob(str(fp)))
		assert len(fps) > 0, f'No data found at {fp}'
		print(f'Found {len(fps)} files in {fp}')
		# assert len(fps) == (20 if self.cfg.task == 'mt80' else 4), \
			# f'Expected 20 files for mt80 task set, 4 files for mt30 task set, found {len(fps)} files.'
	
		# Create buffer for sampling
		_cfg = deepcopy(self.cfg)
		if _cfg.task.startswith("mt"):
			_cfg.episode_length = 101 if self.cfg.task == 'mt80' else 501
			_cfg.buffer_size = 550_450_000 if self.cfg.task == 'mt80' else 345_690_000
			self.buffer = Buffer(_cfg)
			fp = Path(os.path.join(self.cfg.data_dir, '*.pt'))
			fps = sorted(glob(str(fp)))
		elif _cfg.task.startswith("myo"):
			_cfg.episode_length = 101
			_cfg.buffer_size = 505_000
			self.buffer = RobomimicBuffer(_cfg)
			fp = Path(self.cfg.data_dir)
			fps = list(Path(self.cfg.data_dir).rglob('*.hdf5'))
		elif _cfg.task.startswith("d3il-sorting"):
			_cfg.buffer_size = 112128
			self.buffer = D3ILBuffer(_cfg)
		_cfg.steps = _cfg.buffer_size
		
		
		assert len(fps) > 0, f'No data found at {fp}'
		print(f'Found {len(fps)} files in {fp}')
		# tasks = TASK_SET[self.cfg.task]
		for fp in tqdm(fps, desc='Loading data'):
			td = torch.load(fp, weights_only=False)
			# assert td.shape[1] == _cfg.episode_length, \
			# 	f'Expected episode length {td.shape[1]} to match config episode length {_cfg.episode_length}, ' \
			# 	f'please double-check your config.'
			# for i in range(len(td)):
			# 	self.buffer.add(td[i])
			self.buffer.add(td)
		# expected_episodes = _cfg.buffer_size // _cfg.episode_length
		expected_episodes = 600
		assert self.buffer.num_eps == expected_episodes, \
			f'Buffer has {self.buffer.num_eps} episodes, expected {expected_episodes} episodes.'

	def train(self):
		"""Train a TD-MPC2 agent."""
		# assert self.cfg.multitask and self.cfg.task in {'mt30', 'mt80'}, \
		# 	'Offline training only supports multitask training with mt30 or mt80 task sets.'
		self._load_dataset()
		
		print(f'Training agent for {self.cfg.steps} iterations...')
		metrics = {}
		for i in range(self.cfg.steps):
			if i % 1000 == 0:
				print("i ", i)
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
					metrics.update(self.eval(i))
					# self.logger.pprint_multitask(metrics, self.cfg)
					if i > 0:
						self.logger.save_agent(self.agent, identifier=f'{i}')
				self.logger.log(metrics, 'pretrain')
			
		self.logger.finish(self.agent)
