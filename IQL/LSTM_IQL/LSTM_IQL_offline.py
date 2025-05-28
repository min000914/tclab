from pathlib import Path
import numpy as np
import torch
from tqdm import trange
import tclab
import wandb
import os
from datetime import datetime
import math
from src.iql import LSTM_ImplicitQLearning
from src.LSTM_policy import LSTMPolicy
from src.value_functions import TwinQ, ValueFunction
from src.util import (
    set_seed, torchify, Log,
    sample_batch,
    save_csv_png,print_dataset_statistics,normalize_dataset,normalize_reward
)
from src.LSTM_RNN_val import lstm_eval_policy

# GPU 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 Using device:", device)

def create_sequences(data, seq_length=20):
    observations, actions, rewards, next_observations, terminals = [], [], [], [], []

    #print(data['terminals'].)
    terminals_idx = np.where(data['terminals'].cpu() == 1)[0].tolist()
    print(terminals_idx)
    #print(len(terminals_idx))
    start_idx = 0
    for t_idx in terminals_idx:
        obs_seq = data['observations'][start_idx:t_idx+1]
        act_seq = data['actions'][start_idx:t_idx+1] 
        rew_seq = data['rewards'][start_idx:t_idx+1]
        next_obs_seq = data['next_observations'][start_idx:t_idx+1]
        term_seq = data['terminals'][start_idx:t_idx+1]
        #print(term_seq.sum())
        for i in range(len(obs_seq) - seq_length+1):
            #print(obs_seq[i:i + seq_length])
            observations.append(obs_seq[i:i + seq_length])
            actions.append(act_seq[i:i + seq_length])
            rewards.append(rew_seq[i:i + seq_length])
            next_observations.append(next_obs_seq[i:i + seq_length])
            terminals.append(term_seq[i:i + seq_length])
        start_idx = t_idx + 1

    dataset = {
        'observations': torch.stack(observations),
        'actions': torch.stack(actions),
        'rewards': torch.stack(rewards),
        'next_observations': torch.stack(next_observations),
        'terminals': torch.stack(terminals)
    }
    return dataset

def get_env_and_dataset(log, dataset_path, simmul, normalizaion, reward_scale=1.0, obs_scale=1.0, act_scale=1.0, act_normalization=False):
    if simmul:
        from tclab import setup
        lab = setup(connected=False)
        env = lab(synced=False)
    else:
        env = tclab.TCLab()
    env.Q1(0)
    env.Q2(0)

    dataset_np = np.load(dataset_path)
    dataset = {k: torchify(v) for k, v in dataset_np.items()}
    reward_min,reward_max=print_dataset_statistics(dataset=dataset)
    adjusted_min = math.floor(reward_min)
    adjusted_max = math.ceil(reward_max)
    print(dataset)
    if normalizaion:
        dataset = normalize_dataset(dataset, obs_scale=obs_scale, act_scale=act_scale, action_norm=act_normalization)
    
    dataset['rewards'] = normalize_reward(dataset['rewards'], REWARD_MIN=adjusted_min,REWARD_MAX=adjusted_max,reward_scale=reward_scale)
    print_dataset_statistics(dataset=dataset)
    print(dataset)
    
    
    log(f"✅ rewards, observations, next_observations, actions 모두 정규화 완료")
    log(f"Loaded dataset with {len(dataset['observations'])} transitions from {dataset_path}")
    return env, dataset


def main(args):
    torch.set_num_threads(1)
    log = Log(Path(args.log_dir) / args.lab_name, vars(args))
    log(f'Log dir: {log.dir}')
    now_str = datetime.now().strftime('%Y%m%d_%H%M')
    eval_log_path = os.path.join(args.eval_log_path, now_str)

    # wandb 초기화
    wandb.init(entity="TCLab", project="TCLab", name=args.lab_name, config=vars(args))
    
    env, dataset = get_env_and_dataset(log, args.dataset_path, simmul=args.simmul, 
                                       reward_scale=args.reward_scale, obs_scale=args.obs_scale,
                                       normalizaion=args.normalization,
                                       act_scale=args.act_scale, act_normalization=args.act_normalization)
    print(dataset['observations'].shape)
    print(dataset['actions'].shape)
    print(dataset['rewards'].shape)
    print(dataset['next_observations'].shape)  
    print(dataset['terminals'].shape)
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    dataset = create_sequences(dataset, seq_length=args.seq_length)
    print(dataset['observations'].shape)
    print(dataset['actions'].shape)
    print(dataset['rewards'].shape)
    print(dataset['next_observations'].shape)  
    print(dataset['terminals'].shape)
    obs_dim = dataset['observations'].shape[-1]
    print(obs_dim,"@@@@@@@@@@@@")
    act_dim = dataset['actions'].shape[-1]
    set_seed(args.seed)

    policy = LSTMPolicy(obs_dim, act_dim, hidden_dim=args.hidden_dim, lstm_layers=args.lstm_layers, normalization=args.act_normalization).to(device)

    iql = LSTM_ImplicitQLearning(
        qf=TwinQ(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden).to(device),
        vf=ValueFunction(obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden).to(device),
        policy=policy,
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=args.learning_rate),
        max_steps=args.n_steps,
        tau=args.tau,
        beta=args.beta,
        alpha=args.alpha,
        discount=args.discount
    )
    def eval_policy():
        all_datas = []
        if args.simmul:
            eval_returns = []
            #for tsp_seed in range(999_999, 999_999 + args.n_eval_episodes):
            for tsp_seed in range(args.n_eval_episodes):
                tsp_returns = []
                for run_seed in range(args.n_eval_seeds):
                    data = lstm_eval_policy(
                        seed=run_seed, env=env, policy=policy,
                        epi_num=tsp_seed,
                        max_episode_steps=args.max_episode_steps,
                        eval_log_path=eval_log_path,
                        obs_scale=args.obs_scale,
                        act_scale=args.act_scale,
                        normalization=args.normalization,
                        seq_length=args.seq_length,
                        act_normalization=args.act_normalization,
                    )
                    all_datas.append(data)
                    tsp_returns.append(data['total_reward']) 
                eval_returns.append(tsp_returns)
            eval_returns = np.array(eval_returns)
        result = {
            'return mean': eval_returns.mean(),
            **{f'mean_by_tsp/{i}': v for i, v in enumerate(eval_returns.mean(axis=1))},
            **{f'min_by_tsp/{i}': v for i, v in enumerate(eval_returns.min(axis=1))},
            **{f'max_by_tsp/{i}': v for i, v in enumerate(eval_returns.max(axis=1))},
        }
        log.row(result)
        wandb.log(result)
        return result, all_datas

    best_return = -99999.0
    patience = args.patience            # 조기 중단 대기 횟수
    min_delta = 1        # 최소 변화량
    no_improvement_steps = 0 # 개선되지 않은 스텝 카운트

    for step in range(args.n_steps):
        batch = sample_batch(dataset, args.batch_size)
        batch = {k: v.to(device) for k, v in batch.items()}
        #print("batch",batch)
        iql.update(**batch)

        if (step + 1) % args.eval_period == 0:
            result, all_data = eval_policy()
            wandb.log({"step": step + 1})

            current_return = result['return mean']
            
            if current_return > best_return + min_delta:
                save_csv_png(all_data, step + 1)
                best_return = current_return
                best_path = log.dir / 'best.pt'
                torch.save(iql.state_dict(), best_path)
                print(f"📈 Best model saved with return {best_return:.2f}")
                wandb.run.summary['best_return'] = best_return
                wandb.save(str(best_path))
                no_improvement_steps = 0  # 개선되었으니 초기화
            else:
                no_improvement_steps += 1
                print(f"No improvement for {no_improvement_steps} steps...")

            # Early Stopping 조건 확인
            if no_improvement_steps >= patience:
                print("⏹️ Early stopping triggered. Training stopped.")
                break



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
    parser.add_argument('--dataset-path', required=True)
    parser.add_argument('--eval-log-path', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--n-hidden', type=int, default=2)
    parser.add_argument('--lstm-layers', type=int, default=2)
    parser.add_argument('--n-steps', type=int, default=10**6)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--seq-length', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--alpha', type=float, default=0.005)
    parser.add_argument('--tau', type=float, default=0.7)
    parser.add_argument('--beta', type=float, default=3.0)
    parser.add_argument('--deterministic-policy', action='store_true')
    parser.add_argument('--eval-period', type=int, default=5000)
    parser.add_argument('--n-eval-episodes', type=int, default=7)
    parser.add_argument('--n-eval-seeds', type=int, default=3)
    parser.add_argument('--max-episode-steps', type=int, default=1000)
    parser.add_argument('--reward-scale',type=int,default=10.0)
    parser.add_argument('--normalization',default=True)
    parser.add_argument('--act-normalization',default=True)
    parser.add_argument('--obs-scale',type=int,default=1.0)
    parser.add_argument('--act-scale',type=int,default=1.0)
    parser.add_argument('--patience', type=int, default=20)
    main(parser.parse_args())
