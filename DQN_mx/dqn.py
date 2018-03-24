#-*- coding: UTF-8 -*-
"""
filename:                      
function:
    date: 2017/8/7                      
  author: 
________                            ____.__                             
\______ \   ____   ____   ____     |    |__| ____   ____ _____    ____  
 |    |  \_/ __ \ /    \ / ___\    |    |  |/    \_/ ___\\__  \  /    \ 
 |    `   \  ___/|   |  / /_/  /\__|    |  |   |  \  \___ / __ \|   |  \
/_______  /\___  |___|  \___  /\________|__|___|  /\___  (____  |___|  /
        \/     \/     \/_____/                  \/     \/     \/     \/    
"""
import gym
import mxnet as mx
from mxnet import gluon, nd, autograd
import random
from collections import deque
import numpy as np

class Q_network(gluon.nn.Block):
    def __init__(self, hidden_size, action_space):
        super(Q_network, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.n
        with self.name_scope():
            self.dense0 = gluon.nn.Dense(hidden_size)
            self.dense1 = gluon.nn.Dense(num_outputs)

    def forward(self, inputs):
        x = nd.relu(self.dense0(inputs))
        Q_value = self.dense1(x)
        return Q_value

class DQN:
    def __init__(self, hidden_size, action_space, init_epsilon, final_epsilon, gamma, replay_size, batch_size, ctx):
        self.ctx = ctx
        self.action_space = action_space
        self.init_epsilon = init_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon = init_epsilon
        self.gamma = gamma
        self.replay_size = replay_size
        self.batch_size = batch_size
        self.replay_buffer = deque()
        self.model = Q_network(hidden_size, action_space)
        self.model.collect_params().initialize(mx.init.Normal(sigma=0.01), ctx=ctx)
        self.optimizer = gluon.Trainer(self.model.collect_params(), 'adam', {'learning_rate': 0.0001})

    def select_action(self, state, is_train=True):
        if is_train:
            # epsilon greedy
            with autograd.record():
                Q_value = self.model(state.as_in_context(self.ctx))
                action = nd.argmax(Q_value, axis=1)
                if nd.random.uniform(0, 1)[0] < self.epsilon:
                    # select other action
                    action = nd.sample_multinomial(nd.ones_like(Q_value)/self.action_space.n)
            self.epsilon -= (self.init_epsilon - self.final_epsilon) / self.replay_size
        else:
            # no epsilon greedy
            Q_value = self.model(state)
            action = nd.argmax(Q_value)
        return action

    def updata_parameters(self, state, action, reward, next_state, done):
        one_hot_actions = nd.zeros(self.action_space.n).as_in_context(self.ctx)
        one_hot_actions[action[0]] = 1
        self.replay_buffer.append((state, one_hot_actions, reward, next_state, done))
        if len(self.replay_buffer) > self.replay_size:
            self.replay_buffer.popleft()
        if len(self.replay_buffer) > self.batch_size:
            # start training
            # get mini_batch
            with autograd.record():
                minibatch = random.sample(self.replay_buffer, self.batch_size)
                state_batch = nd.array([data[0][0].asnumpy() for data in minibatch]).as_in_context(self.ctx)
                one_hot_action_batch = nd.array([data[1].asnumpy() for data in minibatch]).as_in_context(self.ctx)
                reward_batch = nd.array([data[2] for data in minibatch]).as_in_context(self.ctx)
                next_state_batch = nd.array([data[3] for data in minibatch]).as_in_context(self.ctx)

                y_batch = nd.ones_like(reward_batch)
                Q_value_next_state_batch = self.model(next_state_batch) # get Q(s',a')
                Q_value_next_state_max_a_batch = nd.max(Q_value_next_state_batch, axis=1)# # get max(Q(s',a')) according to a'

                Q_value_batch = self.model(state_batch) # get Q(s): (32, 2)
                Q_value_a_batch = Q_value_batch * one_hot_action_batch
                Q_value_a_batch = nd.sum(Q_value_a_batch, axis=1) # get Q(s,a)

                for i in range(self.batch_size):
                    done = minibatch[i][4]
                    if done:
                        y_batch[i] = reward_batch[i]
                    else:
                        y_batch[i] = reward_batch[i]+self.gamma*Q_value_next_state_max_a_batch[i]
                loss = nd.sum(nd.square(y_batch-Q_value_a_batch))
            self.model.collect_params().zero_grad()
            loss.backward()
            grads = [i.grad(self.ctx) for i in self.model.collect_params().values()]
            gluon.utils.clip_global_norm(grads, 40)
            self.optimizer.step(batch_size=self.batch_size)


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
