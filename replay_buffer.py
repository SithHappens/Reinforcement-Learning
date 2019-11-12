import numpy as np
from collections import deque
import random


class ReplayBuffer:

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()

    def add(self, state, action, reward, done, successor_state):
        experience = (state, action, reward, done, successor_state)

        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1

        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def __len__(self):
        return self.count

    def sample(self, batch_size):
        n = batch_size if self.count >= batch_size else self.count
        batch = random.sample(self.buffer, n)

        # return as s_batch, a_batch, ...
        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        #return [np.array(b) for b in zip(*batch)]
        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0