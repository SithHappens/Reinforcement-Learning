#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

if torch.cuda.is_available():
    from torch.cuda import FloatTensor, LongTensor    # TODO use .to(device)
else:
    from torch import FloatTensor, LongTensor

from utils import AgentTools, Episode, EpisodeStep


class NeuralNetwork(nn.Module):
    ''' Neural Network mapping observations to a policy.
    '''
    
    def __init__(self, n_state, n_hidden, n_actions, use_gpu=torch.cuda.is_available()):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_state, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_actions),
        )
        
        if use_gpu:
            self.net.cuda()  # TODO use .to(device)

    def forward(self, x):
        return self.net(x)


class Agent (AgentTools):
    ''' For discrete actions, softmax in last layer.
    '''
    
    def __init__(self, env, net, reward_solved, lr=0.01, writer_comment='-cartpole'):
        super().__init__(env, writer_comment)
        self.net = net  # without final softmax for faster training
        self.reward_solved = reward_solved

        self.objective = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(params=self.net.parameters(), lr=lr)
        self.softmax = nn.Softmax(dim=1)

    def action_probabilities(self, state):
        state_tensor = FloatTensor([state])    # TODO use .to(device)
        action_probs = self.softmax(self.net(state_tensor)).cpu().data.numpy()[0]
        return action_probs

    def exploration_policy(self, state):
        action_probs = self.action_probabilities(state)
        action = np.random.choice(len(action_probs), p=action_probs)
        return action

    def greedy_policy(self, state):
        action_probs = self.action_probabilities(state)
        action = np.argmax(action_probs)
        return action

    def filter_batch(self, batch, percentile):
        rewards = list(map(lambda s: s.reward, batch))
        reward_bound = np.percentile(rewards, percentile)
        reward_mean = float(np.mean(rewards))

        train_observations = []
        train_actions = []

        for episode in batch:
            if episode.reward < reward_bound:
                continue

            train_observations.extend(map(lambda step: step.state, episode.steps))
            train_actions.extend(map(lambda step: step.action, episode.steps))

        return FloatTensor(train_observations), LongTensor(train_actions), reward_bound, reward_mean    # TODO use .to(device)

    def train(self, percentile=70):
        experience = self.collect_experience(policy=self.exploration_policy, batch_size=16)

        for iter_no, batch in enumerate(experience):
            states, actions, reward_bound, reward_mean = self.filter_batch(batch, percentile)

            self.optimizer.zero_grad()
            predicted_actions = self.net(states)
            loss = self.objective(predicted_actions, actions)
            loss.backward()

            self.optimizer.step()

            self.log(iter_no, loss.item(), reward_bound, reward_mean)

            if reward_mean >= self.reward_solved:
                print('Solved')
                return

    def run(self):
        super().run(self.greedy_policy)
    
    
if __name__ == '__main__':
    import gym

    if True:
        env = gym.make('CartPole-v1')
        net = NeuralNetwork(n_state=env.observation_space.shape[0], n_hidden=128, n_actions=env.action_space.n)
        agent = Agent(env, net, reward_solved=200, writer_comment='-cart pole')
        agent.train()
    
    if False:
        # Naive Version, net converges but reward doesn't increase
        from utils import DiscreteOneHotWrapper
        env = DiscreteOneHotWrapper(gym.make('FrozenLake-v0'))
        net = NeuralNetwork(n_state=env.observation_space.shape[0], n_hidden=128, n_actions=env.action_space.n)
        agent = Agent(env, net, reward_solved=0.8, writer_comment='-FrozenLake')
        agent.train()
    
    '''
    env = gym.make('LunarLander-v2')
    net = NeuralNetwork(n_state=env.observation_space.shape[0], n_hidden=128, n_actions=env.action_space.n)
    agent = Agent(env, net, reward_solved=200, writer_comment='-lunar_lander')
    agent.train()
    '''

    for i in range(10):
        agent.run()
