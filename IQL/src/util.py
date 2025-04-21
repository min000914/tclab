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
    print(f'duration {length}: [{name} 설정 정보]')
    while i < length:
        if length == 600: 
            duration = int(np.clip(np.random.normal(240, 50), 80, 400))
        elif length == 900:
            duration = int(np.clip(np.random.normal(360, 75), 120, 600))
        elif length == 1200:
            duration = int(np.clip(np.random.normal(480, 100), 160, 800))
        else:
            duration = 5
        temp = np.random.uniform(25, 65)
        end = min(i + duration, length)
        tsp[i:end] = temp
        '''print(f'  구간: {i:>3} ~ {end - 1:>3}, 목표 온도: {temp:.2f}°C')'''
        i = end
    return tsp

import time
import os
import matplotlib.pyplot as plt
def real_evalutate_policy(seed, env, policy, epi_num, max_episode_steps,
                          eval_log_path,deterministic=True,st_temp = 29.0,
                          sleep_max=1.0):
    set_seed(seed)
    print(f"Episode{epi_num} eval start...")
    os.makedirs(eval_log_path, exist_ok=True)
    env.Q1(0)
    env.Q2(0)
    # 안전 온도 도달까지 대기
    print(f'Check that temperatures are < {st_temp} degC before starting')
    i = 0
    while env.T1 >= st_temp or env.T2 >= st_temp:
        print(f'Time: {i} T1: {env.T1} T2: {env.T2}')
        i += 20
        time.sleep(20)
    Tsp1 = generate_random_tsp(max_episode_steps, 'TSP1')
    Tsp2 = generate_random_tsp(max_episode_steps, 'TSP2')
    tm = np.zeros(max_episode_steps)
    T1 = np.ones(max_episode_steps) * env.T1
    T2 = np.ones(max_episode_steps) * env.T2
    Q1 = np.zeros(max_episode_steps)
    Q2 = np.zeros(max_episode_steps)
    start_time = time.time()
    prev_time = start_time
    dt_error = 0.0
    # Integral error
    ierr1 = 0.0
    ierr2 = 0.0
    # Integral absolute error
    iae = 0.0
    total_reward = 0.
    csv_filename = os.path.join(eval_log_path,f'PID_episode_{epi_num}_data.csv')
    with open(csv_filename, 'w', newline='') as fid:
        writer = csv.writer(fid)
        writer.writerow(['EPI_Num', 'Time', 'Q1', 'Q2', 'T1', 'T2', 'TSP1', 'TSP2'])
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
            
            # Read temperatures in Kelvin 
            T1[i] = env.T1
            T2[i] = env.T2
            obs = np.array([T1[i], T2[i], Tsp1[i], Tsp2[i]], dtype=np.float32)
            with torch.no_grad():
                action = policy.act(torchify(obs), deterministic=deterministic).cpu().numpy()
            Q1[i],Q2[i]=action    
            
            env.Q1(Q1[i])
            env.Q2(Q2[i])
            reward = -np.linalg.norm([T1[i] - Tsp1[i], T2[i] - Tsp2[i]])
            total_reward += reward
            '''print("{:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6}".format(
                    'Time', 'Tsp1', 'T1', 'Q1', 'Tsp2', 'T2', 'Q2', 'IAE'
                ))
            print(('{:6.1f} {:6.2f} {:6.2f} ' + \
                    '{:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f}').format( \
                        tm[i],Tsp1[i],T1[i],Q1[i],Tsp2[i],T2[i],Q2[i],iae))'''
            writer.writerow([
                    epi_num,
                    f"{tm[i]:.2f}",
                    f"{Q1[i]:.2f}",
                    f"{Q2[i]:.2f}",
                    f"{T1[i]:.2f}",
                    f"{T2[i]:.2f}",
                    f"{Tsp1[i]:.2f}",
                    f"{Tsp2[i]:.2f}"
                ])
            
        plt.figure(figsize=(10, 7))
        ax = plt.subplot(2, 1, 1)
        ax.grid()
        plt.plot(tm,Tsp1,'k--',label=r'$T_1$ set point')
        plt.plot(tm,T1,'b.',label=r'$T_1$ measured')
        plt.plot(tm,Tsp2,'k-',label=r'$T_2$ set point')
        plt.plot(tm,T2,'r.',label=r'$T_2$ measured')
        plt.ylabel(r'Temperature ($^oC$)')
        plt.title(f'Episode {epi_num}')
        plt.legend(loc='best')

        ax = plt.subplot(2, 1, 2)
        ax.grid()
        plt.plot(tm,Q1,'b-',label=r'$Q_1$')
        plt.plot(tm,Q2,'r:',label=r'$Q_2$')
        plt.ylabel('Heater Output (%)')
        plt.xlabel('Time (sec)')
        plt.legend(loc='best')

        plt.tight_layout()
        png_filename=os.path.join(eval_log_path,f'PID_episode_{epi_num}_plot.png')
        plt.savefig(png_filename)
        plt.close()
        
        env.Q1(0)
        env.Q2(0)
        print("All episodes finished.")
        return total_reward

import time
import os
import matplotlib.pyplot as plt
def sim_evalutate_policy(seed, env, policy, epi_num, max_episode_steps,
                          eval_log_path,deterministic=True,
                          sleep_max=1.0):

    env.close()
    from tclab import setup
    lab= setup(connected=False)
    env=lab(synced=False)
    set_seed(seed)
    print(f"Episode{epi_num} eval start...")
    os.makedirs(eval_log_path, exist_ok=True)
    env.Q1(0)
    env.Q2(0)
    Tsp1 = generate_random_tsp(max_episode_steps, 'TSP1')
    Tsp2 = generate_random_tsp(max_episode_steps, 'TSP2')
    tm = np.zeros(max_episode_steps)
    T1 = np.ones(max_episode_steps) * env.T1
    T2 = np.ones(max_episode_steps) * env.T2
    Q1 = np.zeros(max_episode_steps)
    Q2 = np.zeros(max_episode_steps)

    total_reward = 0.
    csv_filename = os.path.join(eval_log_path,f'PID_episode_{epi_num}_data.csv')
    with open(csv_filename, 'w', newline='') as fid:
        writer = csv.writer(fid)
        writer.writerow(['EPI_Num', 'Time', 'Q1', 'Q2', 'T1', 'T2', 'TSP1', 'TSP2'])
        for i in range(max_episode_steps):
            sim_time = i * sleep_max
            env.update(t=sim_time)
            tm[i] = sim_time
            
            # Read temperatures in Kelvin 
            T1[i] = env.T1
            T2[i] = env.T2
            obs = np.array([T1[i], T2[i], Tsp1[i], Tsp2[i]], dtype=np.float32)
            with torch.no_grad():
                action = policy.act(torchify(obs), deterministic=deterministic).cpu().numpy()
            Q1[i],Q2[i]=action    
            
            env.Q1(Q1[i])
            env.Q2(Q2[i])
            reward = -np.linalg.norm([T1[i] - Tsp1[i], T2[i] - Tsp2[i]])
            total_reward += reward
            '''print("{:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6}".format(
                    'Time', 'Tsp1', 'T1', 'Q1', 'Tsp2', 'T2', 'Q2', 'IAE'
                ))
            print(('{:6.1f} {:6.2f} {:6.2f} ' + \
                    '{:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f}').format( \
                        tm[i],Tsp1[i],T1[i],Q1[i],Tsp2[i],T2[i],Q2[i],iae))'''
            writer.writerow([
                    epi_num,
                    f"{tm[i]:.2f}",
                    f"{Q1[i]:.2f}",
                    f"{Q2[i]:.2f}",
                    f"{T1[i]:.2f}",
                    f"{T2[i]:.2f}",
                    f"{Tsp1[i]:.2f}",
                    f"{Tsp2[i]:.2f}"
                ])
            
        plt.figure(figsize=(10, 7))
        ax = plt.subplot(2, 1, 1)
        ax.grid()
        plt.plot(tm,Tsp1,'k--',label=r'$T_1$ set point')
        plt.plot(tm,T1,'b.',label=r'$T_1$ measured')
        plt.plot(tm,Tsp2,'k-',label=r'$T_2$ set point')
        plt.plot(tm,T2,'r.',label=r'$T_2$ measured')
        plt.ylabel(r'Temperature ($^oC$)')
        plt.title(f'Episode {epi_num}')
        plt.legend(loc='best')

        ax = plt.subplot(2, 1, 2)
        ax.grid()
        plt.plot(tm,Q1,'b-',label=r'$Q_1$')
        plt.plot(tm,Q2,'r:',label=r'$Q_2$')
        plt.ylabel('Heater Output (%)')
        plt.xlabel('Time (sec)')
        plt.legend(loc='best')

        plt.tight_layout()
        png_filename=os.path.join(eval_log_path,f'PID_episode_{epi_num}_plot.png')
        plt.savefig(png_filename)
        plt.close()
        
        env.Q1(0)
        env.Q2(0)
        print("All episodes finished.")
        return total_reward


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