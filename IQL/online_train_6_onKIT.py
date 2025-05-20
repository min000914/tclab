import os
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import wandb
import tclab
from tqdm import trange
import time
from src.buffer import ReplayBuffer
from src.iql import ImplicitQLearning
from src.policy import GaussianPolicy, DeterministicPolicy
from src.value_functions import TwinQ, ValueFunction
from src.util import (
    generate_random_tsp, set_seed, torchify, Log,
    normalize_reward,save_csv_png,real_evalutate_policy
)

def main(args):
    data_save_root_path=args.save_data_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log = Log(Path(args.log_dir) / args.lab_name, vars(args))
    print(args)
    wandb.init(entity="TCLab", project="TCLab", name=args.lab_name, config=vars(args))
    now_str = datetime.now().strftime('%Y%m%d_%H%M')
    eval_log_path = os.path.join(args.eval_log_path, now_str)

    obs_dim = args.obs_dim
    act_dim = 2
    buffer = ReplayBuffer()
    buffer.load_dataset("/home/minchanggi/code/TCLab/data/PID2MPC/NPZ/online8_delta5T_2.npz")

    policy_cls =GaussianPolicy
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
        print("eval Start")
        all_datas = []
        if args.simmul:
            eval_returns = []
            for tsp_seed in range(98003, 98003 + args.n_eval_episodes):
            #for tsp_seed in range(args.n_eval_episodes):
                tsp_returns = []
                for run_seed in range(args.n_eval_seeds):
                    data = real_evalutate_policy(
                        seed=run_seed, policy=policy,
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

    '''result,all_data= eval_policy()
    best_return = result['return mean']
    save_csv_png(all_data,0)'''
    best_return=-6093
    
    patience = args.patience            # 조기 중단 대기 횟수
    min_delta = 1        # 최소 변화량
    no_improvement_steps = 0 # 개선되지 않은 스텝 카운트
    


    epi_num= args.n_eval_episodes-1
    epi_step = 0
    done = 1.0 
    initial_exp_prob = args.exp_prob  # 시작 확률
    final_exp_prob = 0.03
    st_temp= 30.0
    sleep_max=1.0
    
    for step in range(args.n_steps):
        #print("step", step, "buffersize", len(buffer), "epistep", epi_step)
        if done: #initialize
            print("Train Start")
            env = tclab.TCLab()
            epi_num += 1
            epi_step = 0
            done = 0.0
            set_seed(epi_num)
            Tsp1 = generate_random_tsp(args.max_episode_steps, 'TSP1')
            Tsp2 = generate_random_tsp(args.max_episode_steps, 'TSP2')
            set_seed(args.seed)
            
            dt_error = 0.0
            while env.T1 >= st_temp or env.T2 >= st_temp:
                time.sleep(20)
                print(env.T1, env.T2, st_temp)
            print("done wait")
            tm_list = np.zeros(args.max_episode_steps)
            T1_list = np.ones(args.max_episode_steps) * env.T1
            T2_list = np.ones(args.max_episode_steps) * env.T2
            Q1_list = np.zeros(args.max_episode_steps)
            Q2_list = np.zeros(args.max_episode_steps)
            start_time = time.time()
            prev_time = start_time    
        
        if (epi_step + 1) % args.max_episode_steps == 0:
            done = 1.0

        if epi_step == 0:
            cur_T1, cur_T2 = env.T1, env.T2
            T1_buffer = [cur_T1] * 5
            T2_buffer = [cur_T2] * 5
            dT1 = 0.0
            dT2 = 0.0
        else :
            cur_T1,cur_T2=next_T1,next_T2
            dT1,dT2=next_dT1,next_dT2
            
        T1_list[epi_step] = cur_T1
        T2_list[epi_step] = cur_T2
        
        obs = np.array([cur_T1,Tsp1[epi_step],dT1,cur_T2,Tsp2[epi_step],dT2])


        decay_rate = 0.99
        if (step) % 1000 == 0:
            #decay_step = max(0, step - args.min_buffer_size)
            decay_step = step
            exp_prob = max(initial_exp_prob * (decay_rate ** (decay_step // 1000)), final_exp_prob) 
        
        #if args.static_exp_prob == True:
        with torch.no_grad():
            action = policy.act(torchify(obs), deterministic=False, exp_prob=args.exp_prob, noise=args.noise,sample=False)
        ''' else:
            with torch.no_grad():
                action,_ = policy.act(torchify(obs), deterministic=False, exp_prob=exp_prob, noise=args.noise, sample=False)'''
        action = action.cpu().numpy()        
            
        Q1, Q2 = action
        Q1_list[epi_step] = Q1
        Q2_list[epi_step] = Q2
        env.Q1(Q1)
        env.Q2(Q2)
        
        
        
        sleep = sleep_max - (time.time() - prev_time) - dt_error
        if sleep >= 1e-6:
            time.sleep(sleep - 1e-6)
        else:
            time.sleep(1e-6)
        
        t = time.time()
        dt = t - prev_time
        
        if (sleep>=1e-6):
            dt_error = dt-sleep_max+0.009
        else:
            dt_error = 0.0
        
        prev_time = t
        tm_list[epi_step] = t - start_time
        
        epi_step += 1
        
        next_T1, next_T2 = env.T1, env.T2
        T1_buffer.append(float(cur_T1))
        T2_buffer.append(float(cur_T2))
        T1_buffer.pop(0)
        T2_buffer.pop(0)

        
        #print(f"epi:{epi_num} step:{epi_step} T1:{cur_T1} T2:{cur_T2} nT1:{next_T1} nT2:{next_T2} Q1:{Q1} Q2:{Q2}, Tsp1:{Tsp1[epi_step]}, Tsp2:{Tsp2[epi_step]}")
        if done:
            #print("doneStep", step)
            next_obs = np.array([next_T1, Tsp1[epi_step-1], next_dT1, next_T2,Tsp2[epi_step-1],next_dT2])
            data={
                "path": data_save_root_path,
                "tm": tm_list,
                "Q1": Q1_list,
                "Q2": Q2_list,
                "T1": T1_list,
                "T2": T2_list,
                "Tsp1": Tsp1,
                "Tsp2": Tsp2,
                }
            data=[data]
            save_csv_png(data,step)
            env.Q1(0)
            env.Q2(0)
            env.close()
        else:
            if epi_step < 4:
                next_dT1 = next_T1 - cur_T1
                next_dT2 = next_T2 - cur_T2
            else:
                next_dT1 = next_T1 - T1_buffer[0] 
                next_dT2 = next_T2 - T2_buffer[0]
            next_obs = np.array([next_T1, Tsp1[epi_step], next_dT1, next_T2,Tsp2[epi_step],next_dT2])  

        raw_reward = -np.linalg.norm([next_T1 - Tsp1[epi_step-1], next_T1 - Tsp2[epi_step-1]])
        reward=normalize_reward(raw_reward, reward_scale=float(args.reward_scale))

        buffer.add((obs, action, next_obs, reward , done))

        if len(buffer) > args.min_buffer_size:
            #print("update",len(buffer))
            batch = buffer.sample(args.batch_size)
            iql.update(**batch)
            if (step + 1) % args.eval_period == 0:
                #print(done,"@@@@",epi_step)
                result, all_data = eval_policy()
                wandb.log({"step": step + 1})
                current_return = result['return mean']
                print(f"{step+1} step: {exp_prob:.4f} exp_prob")
                save_csv_png(all_data,step+1)
                if current_return > best_return + min_delta:

                    best_return = current_return
                    best_path = log.dir / 'best.pt'
                    torch.save(iql.state_dict(), best_path)
                    print(f"Best model saved with return {best_return:.2f}")
                    wandb.run.summary['best_return'] = best_return
                    wandb.save(str(best_path))
                    no_improvement_steps = 0
                else:
                    no_improvement_steps += 1
                    print(f"No improvement for {no_improvement_steps} steps...")

                # Early Stopping 조건 확인
                if no_improvement_steps >= patience:
                    print("Early stopping triggered. Training stopped.")
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
    parser.add_argument('--simmul', default=False)
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
    parser.add_argument('--min-buffer-size', type=int, default=12000)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--alpha', type=float, default=0.005)
    parser.add_argument('--tau', type=float, default=0.7)
    parser.add_argument('--beta', type=float, default=3.0)
    parser.add_argument('--eval-period', type=int, default=5000)
    parser.add_argument('--n-eval-episodes', type=int, default=40)
    parser.add_argument('--n-eval-seeds', type=int, default=1)
    parser.add_argument('--reward-scale',default=10.0)
    parser.add_argument('--max-episode-steps', type=int, default=1200)
    parser.add_argument('--noise', type=float, default=10.0)
    parser.add_argument('--exp-prob', type=float, default=0.3)
    parser.add_argument('--sample', default=False)
    parser.add_argument('--static-exp-prob', default=False)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--save-data-path', default="/home/minchanggi/code/TCLab/data/online_train_data/new/")

    main(parser.parse_args())
