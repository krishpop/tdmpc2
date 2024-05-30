import h5py
import torch
import torch.nn.functional as F
from tensordict.tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SliceSampler
import torchvision.transforms.v2 as transforms
import numpy as np


class Buffer():
    """
    Replay buffer for TD-MPC2 training. Based on torchrl.
    Uses CUDA memory if available, and CPU memory otherwise.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._device = torch.device('cuda')
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

    def _reserve_buffer(self, storage, pin_memory=True):
        """
        Reserve a buffer with the given storage.
        """
        return ReplayBuffer(
            storage=storage,
            sampler=self._sampler,
            pin_memory=pin_memory,
            prefetch=1,
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
        storage_device = 'cuda' if 2.5*total_bytes < mem_free else 'cpu'
        print(f'Using {storage_device.upper()} memory for storage.')
        if storage_device != 'cuda':
            pin_memory = False
        else:
            pin_memory = True
        return self._reserve_buffer(
            LazyTensorStorage(self._capacity, device=torch.device(storage_device)),
            pin_memory=pin_memory
        )

    def _to_device(self, *args, device=None):
        if device is None:
            device = self._device
        return (arg.to(device, non_blocking=True) \
            if arg is not None else None for arg in args)

    def _prepare_batch(self, td):
        """
        Prepare a sampled batch for training (post-processing).
        Expects `td` to be a TensorDict with batch size TxB.
        """
        obs = td['obs'] # ['fixed_camera']
        action = td['action'][1:]
        reward = td['reward'][1:].unsqueeze(-1)
        task = td['task'][0] if 'task' in td.keys() else None
        return self._to_device(obs, action, reward, task)

    def add(self, td):
        """Add an episode to the buffer."""
        td['episode'] = torch.ones_like(td['reward'], dtype=torch.int64) * self._num_eps
        if self._num_eps == 0:
            self._buffer = self._init(td)
        self._buffer.extend(td)
        self._num_eps += 1
        return self._num_eps

    def sample(self):
        """Sample a batch of subsequences from the buffer."""
        td = self._buffer.sample().view(-1, self.cfg.horizon+1).permute(1, 0)
        return self._prepare_batch(td)


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
            if "data" not in f:
                grp = f.create_group("data")
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
                        f.create_dataset(f"data/demo_{episode_id}/{key}/{sub_key}", data=sub_value.cpu().numpy())
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

    def load_hdf5(self, path, image_size=64, pad_to_shape=None, task_id=None):
        transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((image_size, image_size)),
                        transforms.PILToTensor()
                    ])
        with h5py.File(path, "r") as f:
            for episode_id in range(len(f["data"])):
                episode_group = f[f"data/demo_{episode_id}"]
                num_samples = episode_group.attrs["num_samples"]
                if num_samples == 0: continue
                episode_td = {}
                for key in episode_group.keys():
                    if isinstance(episode_group[key], h5py.Group):
                        sub_dict = {}
                        for sub_key in episode_group[key].keys():
                            if sub_key == 'fixed_camera':
                                sub_value = torch.stack([transform(img) for img in episode_group[key][sub_key][:]], dim=0).to(self._device)
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
                episode_transitions['task'] = torch.ones_like(episode_transitions['reward'], dtype=torch.int64) * task_id
                self.add(episode_transitions)
