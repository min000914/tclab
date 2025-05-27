
import time
import tclab
import os
import numpy as np
import torch
from util import (
    generate_random_tsp, set_seed, torchify,
    normalize, unnormalize,
    )


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
                action = policy.act(obs, deterministic=True)
            else:
                action = policy.act(torchify(obs), deterministic=True)
        
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
    if lab_num == 5 or lab_num == 8:
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
        
        if lab_num==5 or lab_num == 8:
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
