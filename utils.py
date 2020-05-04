#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from collections import namedtuple
from tensorboardX import SummaryWriter
import torch


Episode = namedtuple('Episode', 'reward steps')
EpisodeStep = namedtuple('EpisodeStep', 'state action')


class AgentTools:

    def __init__(self, env, writer_comment):
        self.env = env
        self.writer = SummaryWriter(comment=writer_comment)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', self.device)
        print()

        #Additional Info when using cuda
        if self.device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

    def __del__(self):
        self.writer.close()

    def log(self, iter_no, loss, reward_bound, reward_mean):
        print('{}: loss={:.3f}, reward_mean={:.1f}, reward_bound={:.1f}'.format(iter_no, loss, reward_mean, reward_bound))
        self.writer.add_scalar('loss', loss, iter_no)
        self.writer.add_scalar('reward_bound', reward_bound, iter_no)
        self.writer.add_scalar('reward_mean', reward_mean, iter_no)

    def collect_experience(self, policy, batch_size):
        ''' Generator, returns infinite batches of Episodes as
            (episode_reward, [(s,a), (s,a), ...])
        '''
        batch = []
        episode_reward = 0
        steps = []
        state = self.env.reset()

        while True:
            action = policy(state)
            next_state, reward, done, _ = self.env.step(action)

            episode_reward += reward

            steps.append( EpisodeStep(state, action) )

            if done:
                batch.append(Episode(episode_reward, steps))

                episode_reward = 0
                steps = []
                next_state = self.env.reset()

                if len(batch) == batch_size:
                    yield batch
                    batch = []

            state = next_state

    def run(self, policy):
        state = self.env.reset()
        self.env.render()
        done = False

        while not done:
            action = policy(state)

            next_state, reward, done, _ = self.env.step(action)

            self.env.render()

        self.env.close()

