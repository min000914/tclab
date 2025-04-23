from collections import deque
import random
from pathlib import Path
import numpy as np
import torch
from tqdm import trange
import tclab
import wandb
import os
from datetime import datetime

from src.iql import ImplicitQLearning
from src.policy import GaussianPolicy, DeterministicPolicy
from src.value_functions import TwinQ, ValueFunction
from src.util import return_range, generate_random_tsp,set_seed, torchify, Log, sample_batch, evaluate_policy, real_evalutate_policy, sim_evalutate_policy
from main import get_env_and_dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        obs, act, rew, next_obs, done = map(np.stack, zip(*samples))
        return {
            'observations': torchify(obs),
            'actions': torchify(act),
            'rewards': torchify(rew),
            'next_observations': torchify(next_obs),
            'terminals': torchify(done),
        }

    def __len__(self):
        return len(self.buffer)

def get_env(simmul):
    if simmul:
        from tclab import setup
        lab = setup(connected=False)
        env = lab(synced=False)
    else:
        env = tclab.TCLab()
    return env




def main_online(args):
    log = Log(Path(args.log_dir) / args.lab_name, vars(args))
    wandb.init(entity="TCLab", project="TCLab", name=args.lab_name, config=vars(args))
    now_str = datetime.now().strftime('%Y%m%d_%H%M')
    eval_log_path = os.path.join(args.eval_log_path, now_str)

    obs_dim = 6  # [T1, T2, TSP1, TSP2, prevQ1, prevQ2, dT1, dT2]
    act_dim = 2
    env = get_env(simmul=args.simmul)
    set_seed(args.seed)
    buffer = ReplayBuffer(capacity=100_000)

    if args.deterministic_policy:
        policy = DeterministicPolicy(obs_dim, act_dim, args.hidden_dim, args.n_hidden).to(device)
    else:
        policy = GaussianPolicy(obs_dim, act_dim, args.hidden_dim, args.n_hidden).to(device)

    iql = ImplicitQLearning(
        qf=TwinQ(obs_dim, act_dim, args.hidden_dim, args.n_hidden).to(device),
        vf=ValueFunction(obs_dim, args.hidden_dim, args.n_hidden).to(device),
        policy=policy,
        optimizer_factory=lambda p: torch.optim.Adam(p, lr=args.learning_rate),
        max_steps=args.n_steps,
        tau=args.tau,
        beta=args.beta,
        alpha=args.alpha,
        discount=args.discount
    )
    
    def eval_policy(step_num):
        if args.simmul:
            eval_returns = []
            for tsp_seed in range(args.n_eval_episodes):
                tsp_returns = []
                for run_seed in range(args.n_eval_seeds):
                    return_i = sim_evalutate_policy(
                        seed=run_seed, env=env, policy=policy,
                        step_num=step_num, epi_num=tsp_seed,
                        max_episode_steps=args.max_episode_steps,
                        eval_log_path=eval_log_path
                    )
                    tsp_returns.append(return_i)
                eval_returns.append(tsp_returns)
            eval_returns = np.array(eval_returns)
        
        mean_all = eval_returns.mean()
        mean_by_tsp = eval_returns.mean(axis=1)
        min_by_tsp = eval_returns.min(axis=1)
        max_by_tsp = eval_returns.max(axis=1)

        result = {
            'steps': step_num,
            'return mean': mean_all,
        }
        result.update({f'mean_by_tsp/{i}': v for i, v in enumerate(mean_by_tsp)})
        result.update({f'min_by_tsp/{i}': v for i, v in enumerate(min_by_tsp)})
        result.update({f'max_by_tsp/{i}': v for i, v in enumerate(max_by_tsp)})

        log.row(result)
        wandb.log(result)
        return result
    
    Q1_prev = Q2_prev = dT1 = dT2 = 0.0
    sleep_max=1.0
    best_return = -float('inf')
    max_episode_steps=args.max_episodes_steps
    
    for step in range(args.n_steps):
        epi_num, epi_step, epi_reward, done = 0,0,0, True
        
        if (epi_step+1) % max_episode_steps == 0:
            done = True
        
        if done:
            if epi_num > 0:
                log(f"episode: {epi_num} is done epi_reard: {epi_reward}")
            epi_num+=1
            epi_step=0
            epi_reward=0
            done=False
            env.close()
            env = get_env(simmul=args.simmul)
            set_seed(epi_num)
            Tsp1 = generate_random_tsp(max_episode_steps, 'TSP1')
            Tsp2 = generate_random_tsp(max_episode_steps, 'TSP2')
        
        
        
        cur_T1,cur_T2=env.T1,env.T2
        obs=np.array([cur_T1,cur_T2, Tsp1[epi_step],Tsp2[epi_step]])
        
        with torch.no_grad():
            action = policy.act(torchify(obs), deterministic=False).cpu().numpy()
        
        Q1, Q2= action
        env.Q1(Q1)
        env.Q2(Q2)        
        sim_time = epi_step * sleep_max
        env.update(t=sim_time)
        
        next_T1,next_T2=env.T1,env.T2
        
        next_obs=np.array([next_T1,next_T2,Tsp1[epi_step],Tsp2[epi_step]])
        
        reward = -np.linalg.norm([next_T1-Tsp1[epi_step],next_T2-Tsp2[epi_step]])
        
        buffer.add((obs,action,reward,next_obs,done))
        
        epi_reward+=reward
        epi_step+=1
        
        if len(buffer) > args.min_batch_size:
            batch = buffer.sample(args.batch_size)
            iql.update(**batch)
            
        if (step+1) % args.eval_period == 0:
            result = eval_policy(step + 1)
            wandb.log({"step": step + 1})
            if result['return mean'] > best_return:
                best_return = result['return mean']
                best_path = log.dir / 'best.pt'
                torch.save(iql.state_dict(), best_path)
                print(f"📈 Best model saved with return {best_return:.2f}")
                wandb.run.summary['best_return'] = best_return
                wandb.save(str(best_path))

    torch.save(iql.state_dict(), log.dir / 'final.pt')
    wandb.finish()
