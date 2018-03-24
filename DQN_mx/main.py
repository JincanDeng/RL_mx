#-*- coding: UTF-8 -*-
"""
filename: main.py
function: use DQN to solve CartPole-v0
    date: 2017/8/7                      
  author: 
________                            ____.__                             
\______ \   ____   ____   ____     |    |__| ____   ____ _____    ____  
 |    |  \_/ __ \ /    \ / ___\    |    |  |/    \_/ ___\\__  \  /    \ 
 |    `   \  ___/|   |  / /_/  /\__|    |  |   |  \  \___ / __ \|   |  \
/_______  /\___  |___|  \___  /\________|__|___|  /\___  (____  |___|  /
        \/     \/     \/_____/                  \/     \/     \/     \/    
"""
from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd, gluon
import argparse, math, os
import gym
from gym import wrappers
import numpy as np

# argument parser
parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--ctx', type=str, default='gpu', help='The context of this experiment')
parser.add_argument('--env_name', type=str, default='CartPole-v0')
# parser.add_argument('--env_name', type=str, default='InvertedPendulum-v1')
parser.add_argument('--gamma', type=float, default=0.9, metavar='G',
                    help='discount factor for reward (default: 0.9)')
parser.add_argument('--init_epsilon', type=float, default=0.5, metavar='G',
                    help='The initial epsilon for epsilon-greedy')
parser.add_argument('--final_epsilon', type=float, default=0.01, metavar='G',
                    help='The final epsilon for epsilon-greedy')
parser.add_argument('--batch_size', type=int, default=32, metavar='G',
                    help='The batch size')
parser.add_argument('--replay_size', type=int, default=10000, metavar='G',
                    help='The batch size')
parser.add_argument('--exploration_end', type=int, default=100, metavar='N',
                    help='number of episodes with noise (default: 100)')
parser.add_argument('--seed', type=int, default=123, metavar='N',
                    help='random seed (default: 123)')
parser.add_argument('--num_steps', type=int, default=300, metavar='N',
                    help='max episode length (default: 300)')
parser.add_argument('--num_episodes', type=int, default=10000, metavar='N',
                    help='number of episodes (default: 10000)')
parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                    help='number of episodes (default: 128)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--ckpt_freq', type=int, default=100,
		            help='model saving frequency')
parser.add_argument('--display', type=bool, default=False,
                    help='display or not')
args = parser.parse_args()

env_name = args.env_name
env = gym.make(env_name)
out_dir = '/home/xinge/can/code/RL_mx/DQN_mx/output/%s-experiment' % env_name
env = wrappers.Monitor(env, out_dir, force=True)
ctx = mx.cpu() if args.ctx=='cpu' else mx.gpu()
if type(env.action_space) == gym.spaces.discrete.Discrete:
    from dqn import DQN
else:
    # from reinforce_discrete import REINFORCE
    raise NotImplementedError()

agent = DQN(args.hidden_size, env.action_space, args.init_epsilon, args.final_epsilon, args.gamma, args.replay_size, args.batch_size, ctx)
for i_episode in range(args.num_episodes):
    state = nd.array([env.reset()])
    rewards = list()
    for i_step in range(args.num_steps):
        env.render()
        action = agent.select_action(state.as_in_context(ctx))
        next_state, reward, done, _ = env.step(int(action.asnumpy()[0]))
        agent.updata_parameters(state, action, reward, next_state, done)
        state = nd.array([next_state])
        rewards.append(reward)
        if done:
            break
    print("Episode: {}, reward: {}".format(i_episode, np.sum(rewards)))
env.close()
"""
░░░░░░░░░▄░░░░░░░░░░░░░░▄░░░░
░░░░░░░░▌▒█░░░░░░░░░░░▄▀▒▌░░░
░░░░░░░░▌▒▒█░░░░░░░░▄▀▒▒▒▐░░░
░░░░░░░▐▄▀▒▒▀▀▀▀▄▄▄▀▒▒▒▒▒▐░░░
░░░░░▄▄▀▒░▒▒▒▒▒▒▒▒▒█▒▒▄█▒▐░░░
░░░▄▀▒▒▒░░░▒▒▒░░░▒▒▒▀██▀▒▌░░░ 
░░▐▒▒▒▄▄▒▒▒▒░░░▒▒▒▒▒▒▒▀▄▒▒▌░░
░░▌░░▌█▀▒▒▒▒▒▄▀█▄▒▒▒▒▒▒▒█▒▐░░
░▐░░░▒▒▒▒▒▒▒▒▌██▀▒▒░░░▒▒▒▀▄▌░
░▌░▒▄██▄▒▒▒▒▒▒▒▒▒░░░░░░▒▒▒▒▌░
▀▒▀▐▄█▄█▌▄░▀▒▒░░░░░░░░░░▒▒▒▐░
▐▒▒▐▀▐▀▒░▄▄▒▄▒▒▒▒▒▒░▒░▒░▒▒▒▒▌
▐▒▒▒▀▀▄▄▒▒▒▄▒▒▒▒▒▒▒▒░▒░▒░▒▒▐░
░▌▒▒▒▒▒▒▀▀▀▒▒▒▒▒▒░▒░▒░▒░▒▒▒▌░
░▐▒▒▒▒▒▒▒▒▒▒▒▒▒▒░▒░▒░▒▒▄▒▒▐░░
░░▀▄▒▒▒▒▒▒▒▒▒▒▒░▒░▒░▒▄▒▒▒▒▌░░
░░░░▀▄▒▒▒▒▒▒▒▒▒▒▄▄▄▀▒▒▒▒▄▀░░░
░░░░░░▀▄▄▄▄▄▄▀▀▀▒▒▒▒▒▄▄▀░░░░░
░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▀▀░░░░░░░░
"""
