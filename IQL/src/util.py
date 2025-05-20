import csv
from datetime import datetime
import json
from pathlib import Path
import random
import string
import sys

import numpy as np
import torch
import torch.nn as nn


DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Squeeze(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(dim=self.dim)
    
def torchify(x):
    return torch.tensor(x, dtype=torch.float32).to(DEFAULT_DEVICE)


def mlp(dims, activation=nn.ReLU, output_activation=None, squeeze_output=False):
    n_dims = len(dims)
    assert n_dims >= 2, 'MLP requires at least two dims (input and output)'

    layers = []
    for i in range(n_dims - 2):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        layers.append(activation())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    if output_activation is not None:
        layers.append(output_activation())
    if squeeze_output:
        assert dims[-1] == 1
        layers.append(Squeeze(-1))
    net = nn.Sequential(*layers)
    net.to(dtype=torch.float32)
    return net


def compute_batched(f, xs):
    return f(torch.cat(xs, dim=0)).split([len(x) for x in xs])


def update_exponential_moving_average(target, source, alpha):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1. - alpha).add_(source_param.data, alpha=alpha)


def torchify(x):
    x = torch.from_numpy(x)
    if x.dtype is torch.float64:
        x = x.float()
    x = x.to(device=DEFAULT_DEVICE)
    return x



def return_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0., 0
    for r, d in zip(dataset['rewards'], dataset['terminals']):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0., 0
    # returns.append(ep_ret)    # incomplete trajectory
    lengths.append(ep_len)      # but still keep track of number of steps
    assert sum(lengths) == len(dataset['rewards'])
    return min(returns), max(returns)


# dataset is a dict, values of which are tensors of same first dimension
def sample_batch(dataset, batch_size):
    k = list(dataset.keys())[0]
    n, device = len(dataset[k]), dataset[k].device
    for v in dataset.values():
        assert len(v) == n, 'Dataset values must have same length'
    indices = torch.randint(low=0, high=n, size=(batch_size,), device=device)
    #print(indices)
    #print("@@@@@@@@2")
    return {k: v[indices] for k, v in dataset.items()}


def evaluate_policy(env, policy, max_episode_steps, deterministic=True):
    obs = env.reset()
    total_reward = 0.
    for _ in range(max_episode_steps):
        with torch.no_grad():
            action = policy.act(torchify(obs), deterministic=deterministic).cpu().numpy()
        next_obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
        else:
            obs = next_obs
    return total_reward


def generate_random_tsp(length, name='TSP'):
    i = 0
    tsp = np.zeros(length)
    #print(f'duration {length}: [{name} 설정 정보]')
    while i < length:
        if length == 600: 
            duration = int(np.clip(np.random.normal(240, 50), 80, 400))
        elif length == 900:
            duration = int(np.clip(np.random.normal(360, 75), 120, 600))
        elif length == 1200:
            duration = int(np.clip(np.random.normal(480, 100), 160, 800))
        else:
            duration = 1
        temp = np.random.uniform(25, 65)
        end = min(i + duration, length)
        tsp[i:end] = temp
        #print("@@@@@@@@@@@@@@@@")
        #print(f'  구간: {i:>3} ~ {end - 1:>3}, 목표 온도: {temp:.2f}°C')
        i = end
    return tsp

import time
import tclab
def real_evalutate_policy(seed, policy, epi_num, max_episode_steps, eval_log_path,
                         st_temp= 29.0, obs_scale=1.0, act_scale=1.0, sleep_max=1.0, normalization=False,device="cuda:0"):
    env = tclab.TCLab()
    set_seed(epi_num)

    env.Q1(0)
    env.Q2(0)
    while env.T1 >= st_temp or env.T2 >= st_temp:
            #print(f'Time: {i} T1: {env.T1} T2: {env.T2}')
            time.sleep(20)
    Tsp1 = generate_random_tsp(max_episode_steps, 'TSP1')
    Tsp2 = generate_random_tsp(max_episode_steps, 'TSP2')
    set_seed(seed)

    tm = np.zeros(max_episode_steps)
    T1 = np.ones(max_episode_steps) * env.T1
    T2 = np.ones(max_episode_steps) * env.T2
    Q1 = np.zeros(max_episode_steps)
    Q2 = np.zeros(max_episode_steps)

    total_reward = 0.0
    dt_error = 0.0
    start_time = time.time()
    prev_time = start_time
    
    for i in range(max_episode_steps):
        sleep = sleep_max - (time.time() - prev_time) - dt_error
        if sleep >= 1e-4:
            time.sleep(sleep - 1e-4)
        else:
            print('exceeded max cycle time by ' + str(abs(sleep)) + ' sec')
            time.sleep(1e-4)

        t = time.time()
        dt = t - prev_time
        if (sleep>=1e-4):
            dt_error = dt-sleep_max+0.009
        else:
            dt_error = 0.0
        prev_time = t
        tm[i] = t - start_time

        T1[i] = env.T1
        T2[i] = env.T2

        if i == 0:
            dT1, dT2 = 0.0, 0.0
        elif i<4:
            dT1, dT2= T1[i]-T1[i-1], T2[i]-T2[i-1]
        else:
            dT1 = T1[i] - T1[i - 4] 
            dT2 = T2[i] - T2[i - 4]


       
        obs = np.array([T1[i], Tsp1[i], dT1, T2[i], Tsp2[i], dT2], dtype=np.float32)

        
        with torch.no_grad():
            if normalization:
                action,_ = policy.act(obs, deterministic=True)
            else:
                action,_ = policy.act(torchify(obs), deterministic=True)
                action = action.cpu().numpy()
        
        Q1[i], Q2[i] = action

        env.Q1(Q1[i])
        env.Q2(Q2[i])

        reward = -np.linalg.norm([T1[i] - Tsp1[i], T2[i] - Tsp2[i]])
        total_reward += reward

    env.Q1(0)
    env.Q2(0)
    env.close()

    # 결과 저장
    path = os.path.join(eval_log_path, f"{seed}seed", f"{epi_num}epi")
    return {
        "path": path,
        "tm": tm,
        "Q1": Q1,
        "Q2": Q2,
        "T1": T1,
        "T2": T2,
        "Tsp1": Tsp1,
        "Tsp2": Tsp2,
        "total_reward": total_reward,
    }



import os
def sim_evalutate_policy(seed, env, policy, epi_num, max_episode_steps, eval_log_path,
                         obs_scale=1.0, act_scale=1.0, sleep_max=1.0, normalization=False
                         ,action_norm=False,lab_num=5,device="cuda:0"):
    env.close()
    from tclab import setup
    lab = setup(connected=False)
    env = lab(synced=False)

    set_seed(epi_num)

    env.Q1(0)
    env.Q2(0)
    Tsp1 = generate_random_tsp(max_episode_steps, 'TSP1')
    Tsp2 = generate_random_tsp(max_episode_steps, 'TSP2')
    set_seed(seed)

    tm = np.zeros(max_episode_steps)
    T1 = np.ones(max_episode_steps) * env.T1
    T2 = np.ones(max_episode_steps) * env.T2
    Q1 = np.zeros(max_episode_steps)
    Q2 = np.zeros(max_episode_steps)

    total_reward = 0.0
    if lab_num == 5:
        obs_mins = torch.tensor([23.0, 25.0, -1.3, 23.0, 25.0, -1.3], device=device)
        obs_maxs = torch.tensor([67.0, 65.0, 1.3, 67.0, 65.0, 1.3], device=device)
    else: 
        obs_mins = torch.tensor([24.0, 25.0, 24.0, 25.0], device=device)
        obs_maxs = torch.tensor([66.0, 65.0, 66.0, 65.0], device=device)
    act_mins = torch.tensor([0.0, 0.0], device=device)
    act_maxs = torch.tensor([100.0, 100.0], device=device)

    for i in range(max_episode_steps):
        sim_time = i * sleep_max
        env.update(t=sim_time)
        tm[i] = sim_time

        T1[i] = env.T1
        T2[i] = env.T2

        if i == 0:
            dT1, dT2, prevQ1, prevQ2 = 0.0, 0.0, 29.0, 29.0
        elif i<4:
            dT1, dT2, prevQ1, prevQ2 = T1[i]-T1[i-1], T2[i]-T2[i-1], 29.0, 29.0
        else:
            dT1 = T1[i] - T1[i - 4] 
            dT2 = T2[i] - T2[i - 4]
            prevQ1 = Q1[i - 1]
            prevQ2 = Q2[i - 1]
        
        if lab_num==5:
            obs = np.array([T1[i], Tsp1[i], dT1, T2[i], Tsp2[i], dT2], dtype=np.float32)
        #obs = np.array([T1[i], T2[i], Tsp1[i], Tsp2[i], prevQ1,prevQ2,dT1,dT2], dtype=np.float32)
        else:
            obs = np.array([T1[i], Tsp1[i], T2[i],Tsp2[i]], dtype=np.float32)
        if normalization:
            obs = normalize(
                obs,
                min_val=obs_mins,  # 학습 때 사용한 min
                max_val=obs_maxs,  # 학습 때 사용한 max
                scale=obs_scale,
                mode='zero_one'    # obs는 0~1 정규화
            )

        
        with torch.no_grad():
            if normalization:
                action = policy.act(obs, deterministic=True)
            else:
                action = policy.act(torchify(obs), deterministic=True)
        
        action = action.cpu().numpy()
        
        #print(f"action: {action}")
        if action_norm:
            action = unnormalize(
                action,
                min_val=act_mins,
                max_val=act_maxs,
                scale=act_scale,
                mode='zero_one'
            )
        
        Q1[i], Q2[i] = action

        env.Q1(Q1[i])
        env.Q2(Q2[i])

        reward = -np.linalg.norm([T1[i] - Tsp1[i], T2[i] - Tsp2[i]])
        total_reward += reward

    env.Q1(0)
    env.Q2(0)

    # 결과 저장
    path = os.path.join(eval_log_path, f"{seed}seed", f"{epi_num}epi")
    return {
        "path": path,
        "tm": tm,
        "Q1": Q1,
        "Q2": Q2,
        "T1": T1,
        "T2": T2,
        "Tsp1": Tsp1,
        "Tsp2": Tsp2,
        "total_reward": total_reward,
    }


def save_csv_png(all_data,step_num):
    import csv
    import matplotlib.pyplot as plt
    for data in all_data:
        path = data["path"]
        tm = data["tm"]
        Q1 = data["Q1"]
        Q2 = data["Q2"]
        T1 = data["T1"]
        T2 = data["T2"]
        Tsp1 = data["Tsp1"]
        Tsp2 = data["Tsp2"]
        os.makedirs(path, exist_ok=True)
        csv_filename = os.path.join(path, f'episode_{step_num}_data.csv')
        with open(csv_filename, 'w', newline='') as fid:
            writer = csv.writer(fid)
            writer.writerow(['step_num', 'Time', 'Q1', 'Q2', 'T1', 'T2', 'TSP1', 'TSP2'])
            for i in range(len(tm)):
                writer.writerow([
                    step_num,
                    f"{tm[i]:.2f}", f"{Q1[i]:.2f}", f"{Q2[i]:.2f}",
                    f"{T1[i]:.2f}", f"{T2[i]:.2f}", f"{Tsp1[i]:.2f}", f"{Tsp2[i]:.2f}"
                ])

        plt.figure(figsize=(10, 7))
        ax = plt.subplot(2, 1, 1)
        ax.grid()
        plt.plot(tm, Tsp1, 'k--', label=r'$T_1$ set point')
        plt.plot(tm, T1, 'b.', label=r'$T_1$ measured')
        plt.plot(tm, Tsp2, 'k-', label=r'$T_2$ set point')
        plt.plot(tm, T2, 'r.', label=r'$T_2$ measured')
        plt.ylabel(r'Temperature ($^oC$)')
        plt.title(f'Episode {step_num}')
        plt.legend(loc='best')

        ax = plt.subplot(2, 1, 2)
        ax.grid()
        plt.plot(tm, Q1, 'b-', label=r'$Q_1$')
        plt.plot(tm, Q2, 'r:', label=r'$Q_2$')
        plt.ylabel('Heater Output (%)')
        plt.xlabel('Time (sec)')
        plt.legend(loc='best')

        plt.tight_layout()
        png_filename = os.path.join(path, f'episode_{step_num}_plot.png')
        plt.savefig(png_filename)
        plt.close()

def print_dataset_statistics(dataset):
    obs = dataset['observations']
    next_obs = dataset['next_observations']
    act = dataset['actions']
    rew = dataset['rewards']

    print("=== Dataset Statistics ===")

    print("Observations:")
    obs_min, obs_min_idx = obs.min(dim=0)
    obs_max, obs_max_idx = obs.max(dim=0)
    print(f"  min: {obs_min} at index {obs_min_idx}")
    print(f"  max: {obs_max} at index {obs_max_idx}")

    print("\nNext Observations:")
    next_obs_min, next_obs_min_idx = next_obs.min(dim=0)
    next_obs_max, next_obs_max_idx = next_obs.max(dim=0)
    print(f"  min: {next_obs_min} at index {next_obs_min_idx}")
    print(f"  max: {next_obs_max} at index {next_obs_max_idx}")

    print("\nActions:")
    act_min, act_min_idx = act.min(dim=0)
    act_max, act_max_idx = act.max(dim=0)
    print(f"  min: {act_min} at index {act_min_idx}")
    print(f"  max: {act_max} at index {act_max_idx}")

    print("\nRewards:")
    rew_min = rew.min()
    rew_max = rew.max()
    rew_min_idx = rew.argmin()
    rew_max_idx = rew.argmax()
    print(f"  min: {rew_min} at index {rew_min_idx}")
    print(f"  max: {rew_max} at index {rew_max_idx}")
    
    return rew_min, rew_max

def print_dataset_statistics_numpy(dataset):
    obs = dataset['observations']
    next_obs = dataset['next_observations']
    act = dataset['actions']
    rew = dataset['rewards']

    print("=== Dataset Statistics ===")

    print("Observations:")
    print(f"  min: {np.min(obs, axis=0)}")
    print(f"  max: {np.max(obs, axis=0)}")

    print("\nNext Observations:")
    print(f"  min: {np.min(next_obs, axis=0)}")
    print(f"  max: {np.max(next_obs, axis=0)}")

    print("\nActions:")
    print(f"  min: {np.min(act, axis=0)}")
    print(f"  max: {np.max(act, axis=0)}")

    print("\nRewards:")
    print(f"  min: {np.min(rew)}")
    print(f"  max: {np.max(rew)}")
    
    return np.min(rew), np.max(rew)
    
def normalize_reward(r,REWARD_MIN=-49.0,REWARD_MAX=0.0,reward_scale=5.0):
    # 기본 정규화: [-60, 0] → [-1, 1] → [-5, 5]
    r_norm = 2 * (r - REWARD_MIN) / (REWARD_MAX - REWARD_MIN + 1e-8) - 1
    scaled_reward = r_norm * reward_scale

    return scaled_reward
def normalize(x, min_val, max_val, scale=1.0, mode='zero_one'):
    """
    mode:
    - 'zero_one': [min_val, max_val] → [0, 1] → (× scale)
    - 'minus_one_one': [min_val, max_val] → [-1, 1] → (× scale)
    """
    if not torch.is_tensor(x):
        x = torch.from_numpy(x).to(min_val.device)  # min_val.device로 올려야 device 맞음

    if not torch.is_tensor(min_val):
        min_val = torch.tensor(min_val, device=x.device, dtype=x.dtype)
    if not torch.is_tensor(max_val):
        max_val = torch.tensor(max_val, device=x.device, dtype=x.dtype)
    if mode == 'zero_one':
        x_norm = (x - min_val) / (max_val - min_val + 1e-8)
    elif mode == 'minus_one_one':
        x_norm = 2 * (x - min_val) / (max_val - min_val + 1e-8) - 1

    return x_norm * scale

def unnormalize(x, min_val, max_val, scale=1.0, mode='zero_one'):
    
    # 만약 min_val, max_val가 Tensor라면 Numpy로 변환
    if isinstance(min_val, torch.Tensor):
        min_val = min_val.cpu().numpy()
    if isinstance(max_val, torch.Tensor):
        max_val = max_val.cpu().numpy()

    if mode == 'zero_one':
        x_rescaled = x / scale
        x_orig = x_rescaled * (max_val - min_val) + min_val
    elif mode == 'minus_one_one':
        x_rescaled = x / scale
        x_orig = (x_rescaled + 1) * (max_val - min_val) / 2 + min_val

    return x_orig


def normalize_dataset(dataset, lab_num, obs_scale=1.0, act_scale=1.0, action_norm=False):
    obs = dataset['observations']
    next_obs = dataset['next_observations']
    act = dataset['actions']
    
    if lab_num == 5:
        obs_mins = torch.tensor([24.0, 25.0, -1.3, 24.0, 25.0, -1.3], device=obs.device)
        obs_maxs = torch.tensor([66.0, 65.0, 1.3, 66.0, 65.0, 1.3], device=obs.device)
    else:
        obs_mins = torch.tensor([24.0, 25.0, 24.0, 25.0], device=obs.device)
        obs_maxs = torch.tensor([66.0, 65.0, 66.0, 65.0], device=obs.device)
        
    act_mins = torch.tensor([0.0, 0.0], device=act.device)
    act_maxs = torch.tensor([100.0, 100.0], device=act.device)

    # observations: [0, 1] 정규화 후 scale
    dataset['observations'] = normalize(
        obs,
        min_val=obs_mins,
        max_val=obs_maxs,
        scale=obs_scale,
        mode='zero_one'
    )

    # next_observations: [0, 1] 정규화 후 scale
    dataset['next_observations'] = normalize(
        next_obs,
        min_val=obs_mins,
        max_val=obs_maxs,
        scale=obs_scale,
        mode='zero_one'
    )

    # actions: [0, 1] 정규화 후 scale
    if action_norm:
        dataset['actions'] = normalize(
            act,
            min_val=act_mins,
            max_val=act_maxs,
            scale=act_scale,
            mode='zero_one'
        )

    return dataset


def set_seed(seed, env=None):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if env is not None:
        env.seed(seed)


def _gen_dir_name():
    now_str = datetime.now().strftime('%m-%d-%y_%H.%M.%S')
    rand_str = ''.join(random.choices(string.ascii_lowercase, k=4))
    return f'{now_str}_{rand_str}'

class Log:
    def __init__(self, root_log_dir, cfg_dict,
                 txt_filename='log.txt',
                 csv_filename='progress.csv',
                 cfg_filename='config.json',
                 flush=True):
        self.dir = Path(root_log_dir)/_gen_dir_name()
        self.dir.mkdir(parents=True)
        self.txt_file = open(self.dir/txt_filename, 'w')
        self.csv_file = None
        (self.dir/cfg_filename).write_text(json.dumps(cfg_dict))
        self.txt_filename = txt_filename
        self.csv_filename = csv_filename
        self.cfg_filename = cfg_filename
        self.flush = flush

    def write(self, message, end='\n'):
        now_str = datetime.now().strftime('%H:%M:%S')
        message = f'[{now_str}] ' + message
        for f in [sys.stdout, self.txt_file]:
            print(message, end=end, file=f, flush=self.flush)

    def __call__(self, *args, **kwargs):
        self.write(*args, **kwargs)

    def row(self, dict):
        if self.csv_file is None:
            self.csv_file = open(self.dir/self.csv_filename, 'w', newline='')
            self.csv_writer = csv.DictWriter(self.csv_file, list(dict.keys()))
            self.csv_writer.writeheader()

        self(str(dict))
        self.csv_writer.writerow(dict)
        if self.flush:
            self.csv_file.flush()

    def close(self):
        self.txt_file.close()
        if self.csv_file is not None:
            self.csv_file.close()