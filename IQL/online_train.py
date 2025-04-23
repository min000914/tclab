from collections import deque
import random
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import wandb
import tclab
from tqdm import trange

from src.iql import ImplicitQLearning
from src.policy import GaussianPolicy, DeterministicPolicy
from src.value_functions import TwinQ, ValueFunction
from src.util import (
    return_range, generate_random_tsp, set_seed, torchify, Log,
    sample_batch, evaluate_policy, real_evalutate_policy, sim_evalutate_policy
)
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
        return lab(synced=False)
    return tclab.TCLab()


def main(args):
    log = Log(Path(args.log_dir) / args.lab_name, vars(args))
    wandb.init(entity="TCLab", project="TCLab", name=args.lab_name, config=vars(args))
    now_str = datetime.now().strftime('%Y%m%d_%H%M')
    eval_log_path = os.path.join(args.eval_log_path, now_str)

    obs_dim = 4  # [T1, T2, TSP1, TSP2, prevQ1, prevQ2, dT1, dT2]
    act_dim = 2
    env = get_env(args.simmul)
    set_seed(args.seed)
    buffer = ReplayBuffer()

    policy_cls = DeterministicPolicy if args.deterministic_policy else GaussianPolicy
    policy = policy_cls(obs_dim, act_dim, args.hidden_dim, args.n_hidden).to(device)

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
    if args.offline_model_path is not None:
        model_path = Path(args.offline_model_path)
        if model_path.exists():
            iql.load_state_dict(torch.load(model_path, map_location=device))
            print(f"✅ Loaded offline-trained model from: {model_path}")
        else:
            print(f"⚠️ Given offline model path does not exist: {model_path}")
        
       

        
    def eval_policy(step_num):
        eval_returns = []
        for tsp_seed in range(100_000, 100_000 + args.n_eval_episodes):
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

        result = {
            'steps': step_num,
            'return mean': eval_returns.mean(),
            **{f'mean_by_tsp/{i}': v for i, v in enumerate(eval_returns.mean(axis=1))},
            **{f'min_by_tsp/{i}': v for i, v in enumerate(eval_returns.min(axis=1))},
            **{f'max_by_tsp/{i}': v for i, v in enumerate(eval_returns.max(axis=1))},
        }
        log.row(result)
        wandb.log(result)
        return result

    eval_policy(0)
    set_seed(args.seed)
    epi_num, epi_step, epi_reward, done = 4, 0, 0, True
    best_return = -float('inf')

    set_seed(args.seed)
    for step in range(args.n_steps):
        if done:
            if epi_num > 4:
                log(f"episode: {epi_num} is done epi_reard: {epi_reward}")
            epi_num += 1
            epi_step = 0
            epi_reward = 0
            done = False
            env.close()
            env = get_env(args.simmul)
            set_seed(epi_num)
            Tsp1 = generate_random_tsp(args.max_episode_steps, 'TSP1')
            Tsp2 = generate_random_tsp(args.max_episode_steps, 'TSP2')
            set_seed(args.seed)

        
        if (epi_step + 1) % args.max_episode_steps == 0:
            done = True

        cur_T1, cur_T2 = env.T1, env.T2
        obs = np.array([cur_T1, cur_T2, Tsp1[epi_step], Tsp2[epi_step]])

        with torch.no_grad():
            action = policy.act(torchify(obs), deterministic=False).cpu().numpy()

        Q1, Q2 = action
        env.Q1(Q1)
        env.Q2(Q2)
        env.update(t=epi_step * 1.0)

        next_T1, next_T2 = env.T1, env.T2
        next_obs = np.array([next_T1, next_T2, Tsp1[epi_step], Tsp2[epi_step]])
        reward = -np.linalg.norm([next_T1 - Tsp1[epi_step], next_T2 - Tsp2[epi_step]])

        buffer.add((obs, action, reward, next_obs, done))
        epi_reward += reward
        epi_step += 1

        if len(buffer) > args.min_batch_size:
            batch = buffer.sample(args.batch_size)
            iql.update(**batch)

            if (step + 1) % args.eval_period == 0:
                result = eval_policy(step + 1)
                set_seed(args.seed)
                wandb.log({"step": step + 1})
                if result['return mean'] > best_return:
                    best_return = result['return mean']
                    best_path = log.dir / 'best.pt'
                    torch.save(iql.state_dict(), best_path)
                    print(f"📈 Best model saved with return {best_return:.2f}")
                    wandb.run.summary['best_return'] = best_return
                    wandb.save(str(best_path))

    torch.save(iql.state_dict(), log.dir / 'final.pt')
    wandb.save(str(log.dir / 'final.pt'))
    log.close()
    wandb.finish()
    env.close()


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--lab-name', default="TCLab")
    parser.add_argument('--simmul', default=True)
    parser.add_argument('--log-dir', required=True)
    parser.add_argument('--eval-log-path', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--offline-model-path', required=True)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--n-hidden', type=int, default=2)
    parser.add_argument('--n-steps', type=int, default=10**6)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--min-batch-size', type=int, default=5000)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--alpha', type=float, default=0.005)
    parser.add_argument('--tau', type=float, default=0.7)
    parser.add_argument('--beta', type=float, default=3.0)
    parser.add_argument('--deterministic-policy', action='store_true')
    parser.add_argument('--eval-period', type=int, default=5000)
    parser.add_argument('--n-eval-episodes', type=int, default=5)
    parser.add_argument('--n-eval-seeds', type=int, default=3)
    parser.add_argument('--max-episode-steps', type=int, default=1000)

    main(parser.parse_args())
