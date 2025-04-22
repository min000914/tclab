from pathlib import Path
import numpy as np
import torch
from tqdm import trange
import tclab
import wandb

from src.iql import ImplicitQLearning
from src.policy import GaussianPolicy, DeterministicPolicy
from src.value_functions import TwinQ, ValueFunction
from src.util import return_range, set_seed, Log, sample_batch, evaluate_policy, real_evalutate_policy, sim_evalutate_policy


# GPU 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 Using device:", device)

# torchify 함수 수정
def torchify(x):
    return torch.tensor(x, dtype=torch.float32).to(device)

def get_env_and_dataset(log, dataset_path, simmul):
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

    log(f"Loaded dataset with {len(dataset['observations'])} transitions from {dataset_path}")
    return env, dataset

def main(args):
    torch.set_num_threads(1)
    log = Log(Path(args.log_dir) / args.lab_name, vars(args))
    log(f'Log dir: {log.dir}')

    # wandb 초기화
    wandb.init(entity="TCLab", project="TCLab", name=args.lab_name, config=vars(args))

    env, dataset = get_env_and_dataset(log, args.dataset_path, simmul=args.simmul)
    obs_dim = dataset['observations'].shape[1]
    act_dim = dataset['actions'].shape[1]
    set_seed(args.seed)

    if args.deterministic_policy:
        policy = DeterministicPolicy(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden).to(device)
    else:
        print("GaussianPolicy Ready")
        policy = GaussianPolicy(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden).to(device)

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
                        eval_log_path=args.eval_log_path
                    )
                    tsp_returns.append(return_i)
                eval_returns.append(tsp_returns)
            eval_returns = np.array(eval_returns)
        else:
            eval_returns = np.array([
                real_evalutate_policy(
                    seed=args.seed, env=env, policy=policy,
                    step_num=step_num, epi_num=num,
                    max_episode_steps=args.max_episode_steps,
                    eval_log_path=args.eval_log_path
                ) for num in range(args.n_eval_episodes)
            ])

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

    iql = ImplicitQLearning(
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

    best_return = -99999.0
    for step in trange(args.n_steps):
        batch = sample_batch(dataset, args.batch_size)
        batch = {k: v.to(device) for k, v in batch.items()}
        iql.update(**batch)

        if (step + 1) % args.eval_period == 0:
            result = eval_policy(step)
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
    parser.add_argument('--dataset-path', required=True)
    parser.add_argument('--eval-log-path', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--n-hidden', type=int, default=2)
    parser.add_argument('--n-steps', type=int, default=10**6)
    parser.add_argument('--batch-size', type=int, default=256)
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
