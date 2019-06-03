import numpy as np
import time
import argparse
import logz
import os
import gym
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, MultivariateNormal

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

class Policy(nn.Module):
    def __init__(self, env, discrete, n_layers, size):
        super(Policy, self).__init__()
        
        assert n_layers > 0, "Number of hidder layers should be > 0"
        assert size > 0, "Hidden layer size should be > 0"
        self.n_layers = n_layers
        self.size = size
        
        self.env = env
        self.discrete = discrete
        self.log_std = None
        self.ob_dim = self.env.observation_space.shape[0]
        
        if self.discrete:
            self.activation = torch.relu
            self.ac_dim = self.env.action_space.n
        else:
            self.activation = torch.tanh
            self.ac_dim = self.env.action_space.shape[0]
            self.log_std = nn.Parameter(torch.rand(self.ac_dim))
        
        self.fc_in = nn.Linear(self.ob_dim, self.size)
        self.hidden = nn.ModuleList([nn.Linear(self.size, self.size) for _ in range(self.n_layers-1)])       
        self.fc_out = nn.Linear(self.size, self.ac_dim)
        
    def forward(self, x):
        x = self.activation(self.fc_in(x))
        for _, l in enumerate(self.hidden):
            x = self.activation(l(x))
        x = self.fc_out(x)
        
        return x, self.log_std
    
class Baseline(nn.Module):
    def __init__(self, env, discrete, n_layers, size):
        super(Baseline, self).__init__()
        
        assert n_layers > 0, "Number of hidder layers should be > 0"
        assert size > 0, "Hidden layer size should be > 0"
        self.n_layers = n_layers
        self.size = size
        
        self.env = env
        self.discrete = discrete
        self.ob_dim = self.env.observation_space.shape[0]
        
        if self.discrete:
            self.activation = torch.relu
        else:
            self.activation = torch.tanh
            
        self.fc_in = nn.Linear(self.ob_dim, self.size)
        self.hidden = nn.ModuleList([nn.Linear(self.size, self.size) for _ in range(self.n_layers-1)])       
        self.fc_out = nn.Linear(self.size, 1)
        
    def forward(self, x):
        x = self.activation(self.fc_in(x))
        for _, l in enumerate(self.hidden):
            x = self.activation(l(x))
        x = self.fc_out(x)
        
        return x

class Agent(object):
    
    def __init__(self, args):
        
        # environment
        self.env = args['env']
        
        # architecture args
        self.discrete = args['discrete']
        self.n_layers = args['n_layers']
        self.size = args['size']
        self.policy = Policy(self.env, self.discrete, self.n_layers, self.size).to(device)
        self.policy_outputs = (None, None)
        self.baseline = args['nn_baseline']
        if self.baseline:
            self.baseline_net = Baseline(self.env, self.discrete, self.n_layers, self.size).to(device)
            self.baseline_optimizer = torch.optim.Adam(self.baseline_net.parameters())
            self.baseline_criterian = nn.MSELoss()
        
        # episode args
        self.max_episode_length = args['max_episode_length']
        self.min_batch_size = args['batch_size']
        self.reward_to_go = args['reward_to_go']
        
        # training_args
        self.gamma = args['gamma']
        self.normalize_advantages = args['normalize_advantages']
        
    def policy_forward_pass(self, obs):
        self.policy_outputs = self.policy(obs)
        
    def sample_action(self, obs):
        policy_ins = torch.FloatTensor([obs]).to(device)
        self.policy_forward_pass(policy_ins)
        
        if self.discrete:
            outs, _ = self.policy_outputs
            action_sampler = Categorical(F.softmax(outs[0], dim=-1))
            action = action_sampler.sample()
            log_prob = action_sampler.log_prob(action)
            
            return action.item(), log_prob
        else:
            mean, log_std = self.policy_outputs
            std = torch.exp(log_std)
            action_sampler = MultivariateNormal(mean[0], torch.diag(std))
            action = action_sampler.sample()
            log_prob = action_sampler.log_prob(action)
            
            return action.tolist(), log_prob 
    
    def discounted_rewards(self, rewards):
        dis_rewards = rewards[:]
        for i in range(len(dis_rewards)-2, -1, -1):
            dis_rewards[i] = dis_rewards[i] + self.gamma * dis_rewards[i+1]

        return dis_rewards
    
    def sample_trajectory(self):
        observations, log_probs, rewards = [], [], []
        
        obs = self.env.reset()
        step = 0
        while True:
            step += 1
            act, log_prob = self.sample_action(obs)

            observations.append(obs)
            log_probs.append(-log_prob)
            obs, rew, done, _ = self.env.step(act)    
            rewards.append(rew)
                        
            if done or step >= self.max_episode_length:
                break
        if self.reward_to_go:
            returns = self.discounted_rewards(rewards)
        else:
            returns = np.ones_like(rewards) * np.sum(rewards)
        
        return observations, log_probs, sum(rewards), returns, step
    
    def sample_trajectories(self):
        paths = {}
        observations, log_probs, sum_rewards, returns, ep_lengths = [], [], [], [], []
        timesteps_this_batch = 0
        
        while True:
            obs, l_p, rew, ret, ep_steps = self.sample_trajectory()
            observations.extend(obs)
            log_probs.extend(l_p)
            returns.extend(ret)
            sum_rewards.append(rew)
            ep_lengths.append(ep_steps)
            
            timesteps_this_batch += ep_steps
            if timesteps_this_batch > self.min_batch_size:
                break
                
        paths['observations'] = observations
        paths['log_probs'] = log_probs
        paths['sum_rewards'] = sum_rewards
        paths['returns'] = returns
        paths['ep_lengths'] = ep_steps
        
        return paths, timesteps_this_batch

    def normalize_tensor(self, tensor):
        return (tensor - torch.mean(tensor)) / (torch.std(tensor) + 1e-8)
    
    def get_advantage(self, observations, returns):
        
        if self.baseline:
            self.baseline_pred = self.baseline_net(observations)
            baseline = self.baseline_pred.squeeze(dim=1)
            baseline = self.normalize_tensor(baseline) * torch.std(returns) + torch.mean(returns)            
            advantages = returns - baseline
        else:
            advantages = returns
            
        if self.normalize_advantages:
            return self.normalize_tensor(advantages)
        else:
            return advantages
    
    def update_parameters(self, optimizer, log_probs, re, returns):
        loss = torch.mean(torch.mul(torch.stack(log_probs), returns))
        
        if self.baseline:
            baseline_target = self.normalize_tensor(re).unsqueeze(1)
            baseline_loss = self.baseline_criterian(self.baseline_pred, baseline_target)
            
            self.baseline_optimizer.zero_grad()
            optimizer.zero_grad()
            (loss + baseline_loss).backward()
            self.baseline_optimizer.step()
            optimizer.step()
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        return loss

def setup_logger(logdir, locals_):
    # Configure output directory for logging
    logz.configure_output_dir(logdir)
    # Log experimental parameters
    args = inspect.getargspec(train_PG)[0]
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)
    
def train_PG(exp_name, env_name, n_iter, gamma, min_timesteps_per_batch, 
             max_path_length, learning_rate, reward_to_go, animate, logdir, 
             normalize_advantages, nn_baseline, seed, n_layers, size):
    
    start = time.time()
    env = gym.make(env_name)
    
    setup_logger(logdir, locals())
    
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    max_path_length = max_path_length or env.spec.max_episode_steps
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    
    train_args = {
        # environment
        'env': env,
        
        # architecture args
        'n_layers': n_layers,
        'size': size,
        'discrete': discrete,
        'learning_rate': learning_rate,
        
        # trajectory args
        'render': animate,
        'max_episode_length': max_path_length,
        'batch_size': min_timesteps_per_batch,
        
        # estimated return args
        'gamma': gamma,
        'reward_to_go': reward_to_go,
        'nn_baseline': nn_baseline,
        'normalize_advantages': normalize_advantages
    }
    
    agent = Agent(train_args)
    optimizer = torch.optim.Adam(agent.policy.parameters(), lr=learning_rate)
    
    total_timesteps = 0
    for itr in range(n_iter):    
        paths, timesteps_this_batch = agent.sample_trajectories()

        observations = torch.FloatTensor(paths['observations']).to(device)
        returns = torch.FloatTensor(paths['returns']).to(device)
        log_probs = paths['log_probs']

        advantage_returns = agent.get_advantage(observations, returns)
        loss = agent.update_parameters(optimizer, log_probs, returns, advantage_returns)
        
        
        ####################### Logging #######################
        ep_lengths = paths['ep_lengths']
        rewards = paths['sum_rewards']
        total_timesteps += timesteps_this_batch
        
        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", itr)
        logz.log_tabular("AverageReturn", np.mean(rewards))
        logz.log_tabular("StdReturn", np.std(rewards))
        logz.log_tabular("MaxReturn", np.max(rewards))
        logz.log_tabular("MinReturn", np.min(rewards))
        logz.log_tabular("EpLenMean", np.mean(ep_lengths))
        logz.log_tabular("EpLenStd", np.std(ep_lengths))
        logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        logz.dump_tabular()
        
        if itr%50 == 0:
            print(loss, np.mean(rewards))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.0)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--nn_baseline', '-bl', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=64)
    args = parser.parse_args()
    
    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)
        
    max_path_length = args.ep_len if args.ep_len > 0 else None
    
    for e in range(args.n_experiments):
        seed = args.seed + 10*e
        print("Running experiment with seed {}".format(seed))

        train_PG(exp_name=args.exp_name, env_name=args.env_name, n_iter=args.n_iter, gamma=args.discount, 
        min_timesteps_per_batch=args.batch_size, max_path_length=max_path_length, learning_rate=args.learning_rate,
        reward_to_go=args.reward_to_go, animate=args.render, logdir=os.path.join(logdir, str(seed)), 
        normalize_advantages=not(args.dont_normalize_advantages), nn_baseline=args.nn_baseline, seed=seed, 
        n_layers=args.n_layers, size=args.size)

if __name__ == "__main__":
    main()