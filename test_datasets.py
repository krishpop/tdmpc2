import time
import torch
import tracemalloc
from tensordict.tensordict import TensorDict
from tdmpc2.data.buffer import Buffer
from robomimic.data.dataset import make_dataset
from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer, LazyTensorStorage, SliceSampler
from torchrl.envs import GymEnv
from torch.utils.data import DataLoader

def measure_performance(dataset, num_batches=100):
    start_time = time.time()
    tracemalloc.start()
    
    for i, batch in enumerate(dataset):
        if i >= num_batches:
            break
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    end_time = time.time()
    
    return end_time - start_time, peak / 10**6  # Convert to MB

def create_robomimic_dataset(config, cache_mode):
    config.train.cache_mode = cache_mode
    _, dataset, _ = make_dataset(config)
    return dataset

def create_torchrl_dataset(env_name, num_frames, batch_size):
    env = lambda: GymEnv(env_name)
    collector = SyncDataCollector(env, policy=None, total_frames=num_frames, frames_per_batch=batch_size)
    rb = ReplayBuffer(storage=LazyTensorStorage(num_frames), sampler=SliceSampler(num_slices=4), prefetch=1)
    
    for data in collector:
        rb.extend(data)
    
    return DataLoader(rb, batch_size=batch_size, pin_memory=True)

def create_buffer_dataset(env_name, cfg):
    env = GymEnv(env_name)
    buffer = Buffer(cfg)
    
    # Collect some initial data to populate the buffer
    obs = env.reset()
    done = False
    while not done:
        action = env.rand_step()
        next_obs, reward, done, info = env.step(action)
        td = TensorDict({
            "obs": obs,
            "action": action,
            "reward": reward,
            "next_obs": next_obs,
            "done": done
        }, batch_size=[])
        buffer.add(td)
        obs = next_obs
    
    return buffer


def main():
    # Define your robomimic config here
    robomimic_config = {
        "train": {
            "data": [{"name": "your_dataset_name", "path": "your_dataset_path"}],
            "batch_size": 32,
            "cache_mode": "low_dim",  # or "all"
            "action_keys": ["actions"],
            "all_obs_keys": ["obs"]
        }
    }
    
    # Measure performance for robomimic dataset with low_dim cache mode
    robomimic_dataset_low_dim = create_robomimic_dataset(robomimic_config, cache_mode="low_dim")
    time_low_dim, mem_low_dim = measure_performance(robomimic_dataset_low_dim)
    print(f"Robomimic (low_dim): Time = {time_low_dim:.2f}s, Memory = {mem_low_dim:.2f}MB")
    
    # Measure performance for robomimic dataset with all cache mode
    robomimic_dataset_all = create_robomimic_dataset(robomimic_config, cache_mode="all")
    time_all, mem_all = measure_performance(robomimic_dataset_all)
    print(f"Robomimic (all): Time = {time_all:.2f}s, Memory = {mem_all:.2f}MB")
    
    # Measure performance for TorchRL dataset
    torchrl_dataset = create_torchrl_dataset(env_name="CartPole-v1", num_frames=1000, batch_size=32)
    time_torchrl, mem_torchrl = measure_performance(torchrl_dataset)
    print(f"TorchRL: Time = {time_torchrl:.2f}s, Memory = {mem_torchrl:.2f}MB")

if __name__ == "__main__":
    main()