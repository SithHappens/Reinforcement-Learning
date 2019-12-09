#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from collections import namedtuple
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim


Episode = namedtuple('Episode', 'reward steps')
EpisodeStep = namedtuple('EpisodeStep', 'state action')


class Agent:

    def __init__(self, env, net, lr=0.01, writer_comment='-cartpole'):
        self.env = env

        #self.net = NeuralNetwork(n_state=obs_size, n_hidden=128, n_actions=n_actions)
        self.net = net
        self.objective = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(params=self.net.parameters(), lr=lr)

        self.writer = SummaryWriter(comment=writer_comment)

    def __del__(self):
        self.writer.close()

    def experience_generator(self, batch_size):
        ''' Generator, returns infinite batches of Episodes as 
            (total_reward, [(s,a), (s,a), ...])
        '''
        batch = []
        episode_reward = 0
        episode_steps = []

        state = self.env.reset()

        softmax = nn.Softmax(dim=1)

        while True:
            state_v = torch.FloatTensor([state])
            action_probs = softmax(self.net(state_v)).data.numpy()[0]
            action = np.random.choice(len(action_probs), p=action_probs)

            next_state, reward, done, _ = self.env.step(action)

            episode_reward += reward
            episode_steps.append( EpisodeStep(state, action) )

            if done:
                batch.append( Episode(episode_reward, episode_steps) )

                episode_reward = 0
                episode_steps = []
                next_state = self.env.reset()

                if len(batch) == batch_size:
                    yield batch
                    batch = []

            state = next_state

    def filter_batch(self, batch, percentile):
        rewards = list(map(lambda s: s.reward, batch))
        reward_bound = np.percentile(rewards, percentile)
        reward_mean = float(np.mean(rewards))

        train_obs = []
        train_act = []

        for episode in batch:
            if episode.reward < reward_bound:
                continue
            
            train_obs.extend(map(lambda step: step.state, episode.steps))
            train_act.extend(map(lambda step: step.action, episode.steps))

        return torch.FloatTensor(train_obs), torch.LongTensor(train_act), reward_bound, reward_mean

    def train(self, percentile=70):

        for iter_no, batch in enumerate(self.experience_generator(batch_size=16)):
            states, actions, reward_bound, reward_mean = self.filter_batch(batch, percentile)

            self.optimizer.zero_grad()

            predicted_actions = self.net(states)

            loss = self.objective(predicted_actions, actions)
            loss.backward()

            self.optimizer.step()

            print('{}: loss={:.3f}, reward_mean={:.1f}, reward_bound={:.1f}'.format(iter_no, loss.item(), reward_mean, reward_bound))
            self.writer.add_scalar('loss', loss.item(), iter_no)
            self.writer.add_scalar('reward_bound', reward_bound, iter_no)
            self.writer.add_scalar('reward_mean', reward_mean, iter_no)

            if reward_mean > 199:
                print('Solved')
                return

    def run(self):
        state = self.env.reset()
        self.env.render()
        done = False
        softmax = nn.Softmax(dim=1)

        while not done:
            state_v = torch.FloatTensor([state])
            action_probs = softmax(self.net(state_v)).data.numpy()[0]
            action = np.argmax(action_probs)

            if action is None:
                print('Shit.')

            next_state, reward, done, _ = self.env.step(action)

            self.env.render()
        

class NeuralNetwork(nn.Module):

    ''' Neural Network mapping observations to a policy.
    '''

    def __init__(self, n_state, n_hidden, n_actions):
        super().__init__()

        self.net = nn.Sequential(
                nn.Linear(n_state, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_actions)
                # softmax non-linearity is in the agents code
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    import gym

    env = gym.make('CartPole-v0')
    #env = gym.make('Ant-v2')

    net = NeuralNetwork(n_state=env.observation_space.shape[0], n_hidden=128, n_actions=env.action_space.n)

    agent = Agent(env, net, writer_comment='-ant')
    agent.train()

    for i in range(10):
        agent.run()

    #del(agent)
