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
import os
from src.util import (
    generate_random_tsp, set_seed, torchify, Log,
    normalize, unnormalize,
)

def last_value_hold(seq, seq_length):
    current_length = seq.size(0)
    if current_length >= seq_length:
        return seq.unsqueeze(0)  # (1, Seq, Feature)

    # 마지막 값 반복
    last_value = seq[-1].unsqueeze(0).repeat(seq_length - current_length, 1)
    # print(last_value.shape)
    # print(seq.shape)
    # print(last_value)
    # print(seq)
    padded_seq = torch.cat((seq,last_value), dim=0)
    return padded_seq 


def lstm_eval_policy(seed, env, policy, epi_num, max_episode_steps, eval_log_path,
                         seq_length=20, obs_scale=1.0, act_scale=1.0, sleep_max=1.0, normalization=True, act_normalization=True, device="cuda:0"):
    
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

    obs_mins = np.array([24.0, 25.0, 24.0, 25.0], dtype=np.float32)
    obs_maxs = np.array([66.0, 65.0, 66.0, 65.0], dtype=np.float32)

    act_mins = torch.tensor([0.0, 0.0], device=device)
    act_maxs = torch.tensor([100.0, 100.0], device=device)

    obs_sequence = []
    hidden_state = None
    for i in range(max_episode_steps):
        sim_time = i * sleep_max
        env.update(t=sim_time)
        tm[i] = sim_time

        T1[i] = env.T1
        T2[i] = env.T2

        obs = np.array([T1[i], Tsp1[i], T2[i], Tsp2[i]], dtype=np.float32)

        if normalization:
            obs_normalized = (obs - obs_mins) / (obs_maxs - obs_mins)
        else:
            obs_normalized = obs

        obs_sequence.append(obs_normalized)
        
        #if not normalization:
        obs_tensor = torch.from_numpy(np.array(obs_sequence)).to(device).float()

        if len(obs_sequence) < seq_length:
            obs_tensor = last_value_hold(obs_tensor, seq_length=seq_length)
        else:
            obs_tensor = obs_tensor[-seq_length:]
            
        obs_tensor = obs_tensor.unsqueeze(0) 

        with torch.no_grad():
            action,_ = policy.act(obs_tensor,  deterministic=True)
            #print(action)
            action = action.cpu().numpy()
        
        if act_normalization:
            action = unnormalize(
                action,
                min_val=act_mins,
                max_val=act_maxs,
                scale=act_scale,
                mode='zero_one'
            )#.cpu().numpy()
        
        Q1[i], Q2[i] = action.squeeze(0)
        #print(Q1[i], Q2[i])
        env.Q1(Q1[i])
        env.Q2(Q2[i])

        reward = -np.linalg.norm([T1[i] - Tsp1[i], T2[i] - Tsp2[i]])
        total_reward += reward

    env.Q1(0)
    env.Q2(0)

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
