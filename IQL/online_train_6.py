import os
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import wandb
import tclab
from tqdm import trange

from src.buffer import ReplayBuffer
from src.iql import ImplicitQLearning
from src.policy import GaussianPolicy, DeterministicPolicy
from src.value_functions import TwinQ, ValueFunction
from src.util import (
    generate_random_tsp, set_seed, torchify, Log,
    sim_evalutate_policy,
    normalize_reward,save_csv_png
)

def get_env(simmul):
    if simmul:
        from tclab import setup
        lab = setup(connected=False)
        return lab(synced=False)
    return tclab.TCLab()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log = Log(Path(args.log_dir) / args.lab_name, vars(args))
    print(args)
    wandb.init(entity="TCLab", project="TCLab", name=args.lab_name, config=vars(args))
    now_str = datetime.now().strftime('%Y%m%d_%H%M')
    eval_log_path = os.path.join(args.eval_log_path, now_str)

    obs_dim = args.obs_dim
    act_dim = 2
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
        
       
    def eval_policy():
        all_datas = []
        if args.simmul:
            eval_returns = []
            #for tsp_seed in range(999_999, 999_999 + args.n_eval_episodes):
            for tsp_seed in range(args.n_eval_episodes):
                tsp_returns = []
                for run_seed in range(args.n_eval_seeds):
                    data = sim_evalutate_policy(
                        seed=run_seed, env=env, policy=policy,
                        epi_num=tsp_seed,
                        max_episode_steps=args.max_episode_steps,
                        eval_log_path=eval_log_path,
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
    patience = args.patience            # 조기 중단 대기 횟수
    min_delta = 1        # 최소 변화량
    no_improvement_steps = 0 # 개선되지 않은 스텝 카운트
    save_csv_png(all_data,0)
    
    initial_exp_prob = args.exp_prob  # 시작 확률
    final_exp_prob = 0.002
    for step in range(args.n_steps):
        if done:
            epi_num += 1
            epi_step = 0
            epi_reward = 0
            done = 0.0
            env.close()
            env = get_env(args.simmul)
            set_seed(epi_num)
            Tsp1 = generate_random_tsp(args.max_episode_steps, 'TSP1')
            Tsp2 = generate_random_tsp(args.max_episode_steps, 'TSP2')
            set_seed(args.seed)
            exploration = False
            exp_cnt=0

        
        if (epi_step + 1) % args.max_episode_steps == 0:
            done = 1.0

        if epi_step == 0:
            cur_T1, cur_T2 = env.T1, env.T2
            prev_T1, prev_T2 = cur_T1, cur_T2
            prev_Q1, prev_Q2 = 29, 29
            T1_buffer = [cur_T1] * 5
            T2_buffer = [cur_T2] * 5
            dT1 = 0.0
            dT2 = 0.0
        else :
            cur_T1,cur_T2=next_T1,next_T2
            dT1,dT2=next_dT1,next_dT2
        
        '''if epi_step < 4:
            dT1 = cur_T1 - prev_T1
            dT2 = cur_T2 - prev_T2
        else:
            dT1 = cur_T1 - T1_buffer[0]
            dT2 = cur_T2 - T2_buffer[0]'''
            
            
        
        #obs = np.array([cur_T1, cur_T2, Tsp1[epi_step], Tsp2[epi_step],prev_Q1,prev_Q2,dT1,dT2])
        obs = np.array([cur_T1,Tsp1[epi_step],dT1,cur_T2,Tsp2[epi_step],dT2])
        #obs = np.array([cur_T1,Tsp1[epi_step],cur_T2,Tsp2[epi_step]])
######################################################################

        decay_rate = 0.99
        if (step) % 1000 == 0:
            decay_step = max(0, step - args.min_buffer_size)
            exp_prob = max(initial_exp_prob * (decay_rate ** (decay_step // 1000)), final_exp_prob) 
        
        if args.static_exp_prob == True:
            with torch.no_grad():
                action,_ = policy.act(torchify(obs), deterministic=False, exp_prob=args.exp_prob, noise=args.noise,sample=False)
        else:
            with torch.no_grad():
                action,_ = policy.act(torchify(obs), deterministic=False, exp_prob=exp_prob, noise=args.noise, sample=False)
        action = action.cpu().numpy()        
        '''if len(buffer) < args.min_buffer_size:
            with torch.no_grad():
                action, _ = policy.act(torchify(obs), deterministic=True)
                action = action.cpu().numpy()
                
        elif not exploration:
            if args.static_exp_prob == True and args.sample == False:
                with torch.no_grad():
                    action,exploration = policy.act(torchify(obs), deterministic=False, sample=False)
            elif args.static_exp_prob == True and args.sample == True:
                with torch.no_grad():
                    action,exploration = policy.act(torchify(obs), deterministic=False, sample=True)
            elif args.static_exp_prob == False and args.sample == False:
                with torch.no_grad():
                    action,exploration = policy.act(torchify(obs), deterministic=False, exp_prob=exp_prob, sample=False)
            elif args.static_exp_prob == False and args.sample == True:
                with torch.no_grad():
                    action,exploration = policy.act(torchify(obs), deterministic=False,exp_prob=exp_prob, sample=True)
            action = action.cpu().numpy()
            
        else:
            action = np.array([prev_Q1, prev_Q2])
            exp_cnt += 1
            #print("step",step,":",action)
            if exp_cnt > 3:
                exploration = False
                exp_cnt = 0'''
            
        Q1, Q2 = action
        env.Q1(Q1)
        env.Q2(Q2)
        
        epi_step += 1
        env.update(t=epi_step * 1.0)

        next_T1, next_T2 = env.T1, env.T2
        prev_T1,prev_T2=cur_T1,cur_T2
        T1_buffer.append(float(cur_T1))
        T2_buffer.append(float(cur_T2))
        T1_buffer.pop(0)
        T2_buffer.pop(0)
        prev_Q1,prev_Q2=Q1,Q2
        
        #print(f"epi:{epi_num} step:{epi_step} T1:{cur_T1} T2:{cur_T2} nT1:{next_T1} nT2:{next_T2} Q1:{Q1} Q2:{Q2}, Tsp1:{Tsp1[epi_step]}, Tsp2:{Tsp2[epi_step]}")
        if done:
            next_obs=obs
            reward = 0.0
            next_obs = np.array([next_T1, Tsp1[epi_step-1], next_dT1, next_T2,Tsp2[epi_step-1],next_dT2])
        else:
            if epi_step < 4:
                next_dT1 = next_T1 - cur_T1
                next_dT2 = next_T2 - cur_T2
            else:
                next_dT1 = next_T1 - T1_buffer[0] 
                next_dT2 = next_T2 - T2_buffer[0]
            next_obs = np.array([next_T1, Tsp1[epi_step], next_dT1, next_T2,Tsp2[epi_step],next_dT2])
        #print(next_T1.shape, Tsp1[epi_step].shape, next_dT1.shape, next_T2.shape, Tsp2[epi_step].shape, next_dT2.shape)
        

        raw_reward = -np.linalg.norm([next_T1 - Tsp1[epi_step-1], next_T1 - Tsp2[epi_step-1]])
        reward=normalize_reward(raw_reward, reward_scale=float(args.reward_scale))

        buffer.add((obs, action, next_obs, reward , done))
        epi_reward += reward

        if len(buffer) > args.min_buffer_size:
            batch = buffer.sample(args.batch_size)
            #print(batch['next_observations'].shape) 
            iql.update(**batch)
            #iql.only_policy_update(**batch)
            if (step + 1) % args.eval_period == 0:
                result, all_data = eval_policy()
                wandb.log({"step": step + 1})
                current_return = result['return mean']
                print(f"{step+1} step: {exp_prob:.4f} exp_prob")
                if current_return > best_return + min_delta:
                    save_csv_png(all_data,step+1)
                    best_return = current_return
                    best_path = log.dir / 'best.pt'
                    torch.save(iql.state_dict(), best_path)
                    print(f"📈 Best model saved with return {best_return:.2f}")
                    wandb.run.summary['best_return'] = best_return
                    wandb.save(str(best_path))
                    no_improvement_steps = 0
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
    parser.add_argument('--eval-log-path', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--offline-model-path', required=True)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--n-hidden', type=int, default=2)
    parser.add_argument('--n-steps', type=int, default=10**6)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--obs-dim', type=int, default=6)
    parser.add_argument('--min-buffer-size', type=int, default=24000)
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
    parser.add_argument('--exp-prob', type=float, default=0.3)
    parser.add_argument('--sample', default=False)
    parser.add_argument('--static-exp-prob', default=False)
    parser.add_argument('--patience', type=int, default=30)

    main(parser.parse_args())
