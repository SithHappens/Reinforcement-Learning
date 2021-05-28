import os
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense


class ActorCriticNetwork(keras.Model):

    def __init__(self, n_actions, fc1_dims=1024, fc2_dims=512, name='actor_critic', checkpoint_dir='tmp/actor_critic'):
        super().__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_ac')

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.V = Dense(1, activation=None)
        self.pi = Dense(n_actions, activation='softmax')

    def call(self, state):
        value = self.fc1(state)
        value = self.fc2(value)

        V = self.V(value)
        pi = self.pi(value)

        return V, pi

