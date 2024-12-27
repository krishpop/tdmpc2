import h5py
import json
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tensordict.tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SliceSampler
from envs.myosuite import MYOSUITE_TASKS
import torchvision.transforms.v2 as transforms
import numpy as np


class Buffer():
	"""
	Replay buffer for TD-MPC2 training. Based on torchrl.
	Uses CUDA memory if available, and CPU memory otherwise.
	"""

	def __init__(self, cfg):
		self.cfg = cfg
		self._device = torch.device('cuda:0')
		self._capacity = min(cfg.buffer_size, cfg.steps)
		self._sampler = SliceSampler(
			num_slices=self.cfg.batch_size,
			end_key=None,
			traj_key='episode',
			truncated_key=None,
			strict_length=True,
		)
		self._batch_size = cfg.batch_size * (cfg.horizon+1)
		self._num_eps = 0

	@property
	def capacity(self):
		"""Return the capacity of the buffer."""
		return self._capacity

	@property
	def num_eps(self):
		"""Return the number of episodes in the buffer."""
		return self._num_eps

	def _reserve_buffer(self, storage):
		"""
		Reserve a buffer with the given storage.
		"""
		return ReplayBuffer(
			storage=storage,
			sampler=self._sampler,
			pin_memory=False,
			prefetch=0,
			batch_size=self._batch_size,
		)

	def _init(self, tds):
		"""Initialize the replay buffer. Use the first episode to estimate storage requirements."""
		print(f'Buffer capacity: {self._capacity:,}')
		mem_free, _ = torch.cuda.mem_get_info()
		bytes_per_step = sum([
				(v.numel()*v.element_size() if not isinstance(v, TensorDict) \
				else sum([x.numel()*x.element_size() for x in v.values()])) \
			for v in tds.values()
		]) / len(tds)
		total_bytes = bytes_per_step*self._capacity
		print(f'Storage required: {total_bytes/1e9:.2f} GB')
		# Heuristic: decide whether to use CUDA or CPU memory
		storage_device = 'cuda:0' if 2.5*total_bytes < mem_free else 'cpu'
		print(f'Using {storage_device.upper()} memory for storage.')
		self._storage_device = torch.device(storage_device)
		return self._reserve_buffer(
			LazyTensorStorage(self._capacity, device=self._storage_device)
		)

	def _prepare_batch(self, td):
		"""
		Prepare a sampled batch for training (post-processing).
		Expects `td` to be a TensorDict with batch size TxB.
		"""
		td = td.select("obs", "action", "reward", "task", strict=False).to(self._device, non_blocking=True)
		obs = td.get('obs').contiguous()
		action = td.get('action')[1:].contiguous()
		reward = td.get('reward')[1:].unsqueeze(-1).contiguous()
		task = td.get('task', None)
		if task is not None:
			task = task[0].contiguous()
		return obs, action, reward, task

	def add(self, td):
		"""Add an episode to the buffer."""
		td['episode'] = torch.full_like(td['reward'], self._num_eps, dtype=torch.int64)
		if self._num_eps == 0:
			self._buffer = self._init(td)
		self._buffer.extend(td)
		self._num_eps += 1
		return self._num_eps

	def sample(self):
		"""Sample a batch of subsequences from the buffer."""
		td = self._buffer.sample()
		td = td.view(-1, self.cfg.horizon+1+self.cfg.num_frames-1).permute(1, 0)
		return self._prepare_batch(td)


class D3ILBuffer(Buffer):
	"""
	Replay buffer for TD-MPC2 training. Based on torchrl.
	Uses CUDA memory if available, and CPU memory otherwise.
	"""
	def _init(self, tds):
		"""Initialize the replay buffer. Use the first episode to estimate storage requirements."""
		print(f'Buffer capacity: {self._capacity:,}')
		mem_free, _ = torch.cuda.mem_get_info()
		total_bytes = sum([
				(v.numel()*v.element_size() if not isinstance(v, TensorDict) \
				else sum([x.numel()*x.element_size() for x in v.values()])) \
			for v in tds.values()
		])
		print(f'Storage required: {total_bytes/1e9:.2f} GB')
		# Heuristic: decide whether to use CUDA or CPU memory
		storage_device = 'cuda:0' if 2.5*total_bytes < mem_free else 'cpu'
		print(f'Using {storage_device.upper()} memory for storage.')
		self._storage_device = torch.device(storage_device)
		return self._reserve_buffer(
			LazyTensorStorage(self._capacity, device=self._storage_device)
		)

	def add(self, td):
		"""Add the data to the buffer."""
		assert self._num_eps == 0 # this should only be called once
		self._buffer = self._init(td)
		self._buffer.extend(td['fields'])
		self._num_eps = (td['fields']['episode'].max().item() - td['fields']['episode'].min().item()) + 1
		return self._num_eps


class RobomimicBuffer(Buffer):
	def __init__(self, cfg):
		super().__init__(cfg)
		self.use_buffer = cfg.use_buffer
		self.hdf5_path = cfg.hdf5_path
		if self.use_buffer:
			self._buffer = None

	def get_episode(self, episode_id):
		"""Retrieve a TensorDict for a specific episode."""
		return self._buffer.storage[self._buffer.storage["episode"] == episode_id]

	def save_hdf5(self, episode_td): 
		episode_id = self._num_eps - 1
		with h5py.File(self.hdf5_path, "a") as f:
			task_keys = list(MYOSUITE_TASKS.keys())
			if "data" not in f:
				grp = f.create_group("data")
				env_kwargs = OmegaConf.to_container(self.cfg, resolve=True)
				env_kwargs["pad_to_shape"] = (115,)
				env_args = {"env_name": MYOSUITE_TASKS[self.cfg.task], "type": 4, "env_kwargs": env_kwargs}
				f["data"].attrs["env_args"] = json.dumps(env_args)                
			else:
				grp = f["data"]
				episode_id = len(grp.keys())

			# Convert TensorDict to numpy and save to HDF5
			f.create_group(f"data/demo_{episode_id}")
			f[f"data/demo_{episode_id}"].attrs["num_samples"] = len(episode_td)
			for key, value in episode_td.items():
				if isinstance(value, TensorDict):
					f.create_group(f"data/demo_{episode_id}/{key}")
					for sub_key, sub_value in value.items():
						if len(sub_value.shape) == 4 and sub_value.shape[-1] != 3:
							sub_value = sub_value.permute(0, 2, 3, 1)  # Convert (B, C, H, W) to (B, H, W, C)
						f.create_dataset(f"data/demo_{episode_id}/{key}/{sub_key}", data=sub_value.cpu().numpy())
					if key == "obs":
						f.create_dataset(f"data/demo_{episode_id}/{key}/task_id", data=np.repeat(task_keys.index(self.cfg.task), len(value["vec_obs"])))
				else:
					f.create_dataset(f"data/demo_{episode_id}/{key}", data=value.cpu().numpy())
			return len(grp.keys())

	def add(self, td):
		"""Add an episode to the buffer and optionally save to HDF5."""
		td['episode'] = torch.ones_like(td['reward'], dtype=torch.int64) * self._num_eps
		if self.use_buffer:
			if self._num_eps == 0:
				self._buffer = self._init(td)
			self._buffer.extend(td)
		self._num_eps += 1

		if self.hdf5_path:
			if self.use_buffer:
				episode_td = self.get_episode(self._num_eps - 1)
			else:
				episode_td = td
			total_episodes = self.save_hdf5(episode_td)
			return total_episodes

		return self._num_eps

	def load_hdf5(self, path, render_size=64, pad_to_shape=None, task_id=None, image_key='fixed_camera', num_episode_limit = None):
		pil_to_tensor = [
					transforms.ToPILImage(),
					transforms.Resize((render_size, render_size)),
					transforms.PILToTensor()
				]
		transform = transforms.Compose(pil_to_tensor)

		with h5py.File(path, "r") as f:
			num_episodes_loaded = 0
			for episode_id in range(len(f["data"])):
				episode_group = f[f"data/demo_{episode_id}"]
				num_samples = episode_group.attrs["num_samples"]
				if num_samples == 0: continue
				episode_td = {}
				if len(f[f'data/demo_{episode_id}/obs/vec_obs']) != num_samples:
					print(f"skipping demo_{episode_id} because vec_obs length mismatch")
					continue
				for key in episode_group.keys():
					if isinstance(episode_group[key], h5py.Group):
						sub_dict = {}
						for sub_key in episode_group[key].keys():
							if sub_key == image_key: 
								imgs = episode_group[key][sub_key][:]
								if imgs.shape[-1] != 3:
									imgs = imgs.transpose(0, 2, 3, 1)
								sub_value = torch.stack([transform(img) for img in imgs], dim=0).to(self._device)
								if self.cfg.multitask:
									sub_key = 'rgb'
							elif sub_key == 'vec_obs' and pad_to_shape is not None:
								sub_value = episode_group[key][sub_key][:]
								sub_value = torch.tensor(np.pad(sub_value, pad_width=[(0,0), (0, pad_to_shape - sub_value.shape[1])]), device=self._device)
								if self.cfg.multitask:
									sub_key = 'state'
							else:
								sub_value = torch.tensor(episode_group[key][sub_key][:], device=self._device)
							sub_dict[sub_key] = sub_value
						episode_td[key] = TensorDict(sub_dict, batch_size=(num_samples,), device=self._device)
					else:
						episode_td[key] = torch.tensor(episode_group[key][:], device=self._device)
				episode_transitions = TensorDict(episode_td, batch_size=(num_samples,))
				if task_id is not None:
					episode_transitions['task'] = torch.ones_like(episode_transitions['reward'], dtype=torch.int64) * task_id
				self.add(episode_transitions)
				num_episodes_loaded += 1
				if num_episode_limit is not None and num_episodes_loaded > num_episode_limit:
					break
