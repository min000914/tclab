import os
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import wandb
import tclab
from tqdm import trange
from src.iql import LSTM_ImplicitQLearning
from src.LSTM_policy import LSTMPolicy
from src.buffer import LSTMReplayBuffer
from src.iql import ImplicitQLearning
from src.value_functions import TwinQ, ValueFunction
from src.util import (
    generate_random_tsp, set_seed, torchify, Log,
    normalize,unnormalize,
    normalize_reward,save_csv_png
)
from src.LSTM_RNN_val import lstm_eval_policy

def get_env(simmul):
    if simmul:
        from tclab import setup
        lab = setup(connected=False)
        return lab(synced=False)
    return tclab.TCLab()

def last_value_hold(seq, seq_length):
    print(seq.shape)
    print(seq)
    current_length = seq.size(0)
    if current_length >= seq_length:
        return seq.unsqueeze(0)  # (1, Seq, Feature)

    # 마지막 값 반복
    last_value = seq[-1].unsqueeze(0).repeat(seq_length - current_length, 1)
    # print(last_value.shape)
    # print(seq.shape)
    # print(last_value)
    # print(seq)
    padded_seq = torch.cat((last_value, seq), dim=0)
    return padded_seq.unsqueeze(0)  # (1, Seq, Feature)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log = Log(Path(args.log_dir) / args.lab_name, vars(args))
    print(args)
    wandb.init(entity="TCLab", project="TCLab", name=args.lab_name, config=vars(args))
    now_str = datetime.now().strftime('%Y%m%d_%H%M')
    eval_log_path = os.path.join(args.eval_log_path, now_str)
    obs_mins = np.array([24.0, 25.0, 24.0, 25.0], dtype=np.float32)
    obs_maxs = np.array([66.0, 65.0, 66.0, 65.0], dtype=np.float32)

    act_mins = torch.tensor([0.0, 0.0], device=device)
    act_maxs = torch.tensor([100.0, 100.0], device=device)
    obs_scale=1.0
    act_scale=1.0
    obs_dim = args.obs_dim  
    act_dim = 2
    buffer = LSTMReplayBuffer()

    policy = LSTMPolicy(obs_dim, act_dim, hidden_dim=args.hidden_dim, lstm_layers=args.lstm_layers).to(device)

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

    env = get_env(args.simmul)
    epi_num, epi_step, epi_reward, done = args.n_eval_episodes-1, 0, 0, 1.0
    result,all_data= eval_policy()
    best_return = result['return mean']
    save_csv_png(all_data,0)
    
    initial_bias_prob = args.bias_prob  # 시작 확률
    final_bias_prob = 0.05
    for step in range(args.n_steps):
        if done:
            #if epi_num > 0:
                #log(f"episode: {epi_num} is done epi_reard: {epi_reward}")
            epi_num += 1
            epi_step = 0
            epi_reward = 0
            done = 0.0
            obs_sequence = []
            env.close()
            env = get_env(args.simmul)
            set_seed(epi_num)
            #print(epi_num,"!!!!!!!!!!!!!!")
            Tsp1 = generate_random_tsp(args.max_episode_steps, 'TSP1')
            Tsp2 = generate_random_tsp(args.max_episode_steps, 'TSP2')
            set_seed(args.seed)
            
            obs_sequence = []
            action_sequence = []
            next_obs_sequence = []
            reward_sequence = []
            done_sequence = []

        
        if (epi_step + 1) % args.max_episode_steps == 0:
            done = 1.0

        if epi_step == 0:
            cur_T1, cur_T2 = env.T1, env.T2
        else :
            cur_T1,cur_T2=next_T1,next_T2

        
        obs = np.array([cur_T1,Tsp1[epi_step],cur_T2,Tsp2[epi_step]])
        if args.normalization:
            obs_tensor = normalize(
                obs,
                min_val=obs_mins,  
                max_val=obs_maxs,  
                scale=obs_scale,
                mode='zero_one'   
            ).unsqueeze(0)
        '''decay_rate = 0.99
        if (step) % 1000 == 0:
            bias_prob = max(initial_bias_prob * (decay_rate ** (step// 1000)), final_bias_prob) 
        '''
        obs_sequence.append(obs)
        if len(obs_sequence) < args.seq_length:
            obs = last_value_hold(obs_sequence, seq_length=args.seq_length)
        else:
            obs = obs[-args.seq_length:].unsqueeze(0)
            
        with torch.no_grad():
            action,_ = policy.act(obs,  deterministic=True)
            action = action.cpu().numpy()
        if args.normalization:
            action = unnormalize(
                action,
                min_val=act_mins,
                max_val=act_maxs,
                scale=act_scale,
                mode='zero_one'
            )#.cpu().numpy()
        Q1, Q2 = action
        env.Q1(Q1)
        env.Q2(Q2)
        
        epi_step += 1
        env.update(t=epi_step * 1.0)

        next_T1, next_T2 = env.T1, env.T2

        #print(f"epi:{epi_num} step:{epi_step} T1:{cur_T1} T2:{cur_T2} nT1:{next_T1} nT2:{next_T2} Q1:{Q1} Q2:{Q2}, Tsp1:{Tsp1[epi_step]}, Tsp2:{Tsp2[epi_step]}")
        if done:
            next_obs=obs
            reward = 0.0 
        else:
            next_obs = np.array([next_T1, Tsp1[epi_step], next_T2, Tsp2[epi_step]])
            raw_reward = -np.linalg.norm([next_T1 - Tsp1[epi_step], next_T2 - Tsp2[epi_step]])
            reward=normalize_reward(raw_reward, reward_scale=float(args.reward_scale))
        
        if args.normalization:
            action = normalize(
                action,
                min_val=act_mins,  
                max_val=act_maxs,  
                scale=act_scale,
                mode='zero_one'   
            ).unsqueeze(0)
        obs_sequence.append(obs)
        action_sequence.append(action)
        next_obs_sequence.append(next_obs)
        reward_sequence.append(reward)
        done_sequence.append(done)
        
        buffer.add((obs, action, next_obs, reward , done))
        epi_reward += reward

        if len(buffer) > args.min_batch_size:
            batch = buffer.sample(args.batch_size)
            #print(batch['next_observations'].shape) 
            iql.update(**batch)

            if (step + 1) % args.eval_period == 0:
                result, all_data = eval_policy()
                wandb.log({"step": step + 1})
                print(f"{step+1} step: {bias_prob:.4f} bias_prob")
                if result['return mean'] > best_return:
                    save_csv_png(all_data,step+1)
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
    parser.add_argument('--lstm-layers', type=int, default=2)
    parser.add_argument('--seq-length', type=int, default=10)
    parser.add_argument('--normalization',default=False)
    parser.add_argument('--obs-scale', type=float, default=1.0)
    parser.add_argument('--act-scale', type=float, default=1.0)
    parser.add_argument('--n-steps', type=int, default=10**6)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--obs-dim', type=int, default=4)
    parser.add_argument('--min-batch-size', type=int, default=54900)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--alpha', type=float, default=0.005)
    parser.add_argument('--tau', type=float, default=0.7)
    parser.add_argument('--beta', type=float, default=3.0)
    parser.add_argument('--deterministic-policy', action='store_true')
    parser.add_argument('--eval-period', type=int, default=5000)
    parser.add_argument('--n-eval-episodes', type=int, default=7)
    parser.add_argument('--n-eval-seeds', type=int, default=3)
    parser.add_argument('--reward-scale',default=10.0)
    parser.add_argument('--max-episode-steps', type=int, default=1000)
    parser.add_argument('--noise', type=float, default=3.5)
    parser.add_argument('--bias-prob', type=float, default=0.25)
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--static-bias-prob', default=False)

    main(parser.parse_args())
