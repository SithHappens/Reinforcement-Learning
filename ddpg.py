import os
import numpy as np
import tensorflow as tf
from tensorflow.initializers import random_uniform
from replay_buffer import ReplayBuffer
from noise import OUActionNoise


class Actor:

    def __init__(self, learning_rate, input_dims, n_actions, name, sess, action_bound, batch_size=64, chkpt_dir='tmp/ddpg'):
        self.alpha = learning_rate
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.name = name
        self.sess = sess
        self.action_bound = action_bound
        self.batch_size = batch_size
        self.chkpt_dir = chkpt_dir

        self.build_network()

        self.params = tf.trainable_variables(scope=self.name)  # each model should be updated in its own scope
        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(self.chkpt_dir, name+'_ddpg.chkpt')

        self.unnormalized_actor_gradients = tf.gradients(self.mu, self.params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        self.optimize = tf.train.AdamOptimizer(self.alpha).apply_gradients(zip(self.actor_gradients, self.params))

    def build_network(self):
        fc1_dims = 400
        fc2_dims = 300

        with tf.variable_scope(self.name):
            self.input = tf.placeholder(tf.float32, shape=[None, *self.input_dims], name='inputs')
            self.action_gradient = tf.placeholder(tf.float32, shape=[None, self.n_actions])

            f1 = 1 / np.sqrt(fc1_dims)
            dense1 = tf.layers.dense(self.input, units=fc1_dims, kernel_initializer=random_uniform(-f1, f1), bias_initializer=random_uniform(-f1, f1))
            batch1 = tf.layers.batch_normalization(dense1)
            layer1_activation = tf.nn.relu(batch1)

            f2 = 1 / np.sqrt(fc2_dims)
            dense2 = tf.layers.dense(layer1_activation, units=fc2_dims, kernel_initializer=random_uniform(-f2, f2), bias_initializer=random_uniform(-f2, f2))
            batch2 = tf.layers.batch_normalization(dense2)
            layer2_activation = tf.nn.relu(batch2)

            f3 = 0.003
            mu = tf.layers.dense(layer2_activation, units=self.n_actions, activation='tanh', kernel_initializer=random_uniform(-f3, f3), bias_initializer=random_uniform(-f3, f3))
            self.mu = tf.multiply(mu, self.action_bound)

    def predict(self, inputs):
        return self.sess.run(self.mu, feed_dict={self.input: inputs})

    def train(self, inputs, gradients):
        self.sess.run(self.optimize, feed_dict={self.input: inputs, self.action_gradient: gradients})

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        self.saver.save(self.sess, self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.saver.restore(self.sess, self.checkpoint_file)


class Critic:

    def __init__(self, learning_rate, input_dims, n_actions, name, sess, batch_size=64, chkpt_dir='tmp/ddpg'):
        self.alpha = learning_rate
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.name = name
        self.sess = sess
        self.batch_size = batch_size
        self.chkpt_dir = chkpt_dir

        self.build_network()

        self.params = tf.trainable_variables(scope=self.name)  # each model should be updated in its own scope
        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(self.chkpt_dir, name+'_ddpg.chkpt')

        self.optimize = tf.train.AdamOptimizer(self.alpha).minimize(self.loss)

        self.action_gradients = tf.gradients(self.q, self.actions)

    def build_network(self):
        fc1_dims = 400
        fc2_dims = 300

        with tf.variable_scope(self.name):
            self.input = tf.placeholder(tf.float32, shape=[None, *self.input_dims], name='inputs')
            self.actions = tf.placeholder(tf.float32, shape=[None, self.n_actions], name='actions')
            self.q_target = tf.placeholder(tf.float32, shape=[None, 1], name='targets')

            f1 = 1 / np.sqrt(fc1_dims)
            dense1 = tf.layers.dense(self.input, units=fc1_dims, kernel_initializer=random_uniform(-f1, f1), bias_initializer=random_uniform(-f1, f1))
            batch1 = tf.layers.batch_normalization(dense1)
            layer1_activation = tf.nn.relu(batch1)

            f2 = 1 / np.sqrt(fc2_dims)
            dense2 = tf.layers.dense(layer1_activation, units=fc2_dims, kernel_initializer=random_uniform(-f2, f2), bias_initializer=random_uniform(-f2, f2))
            batch2 = tf.layers.batch_normalization(dense2)

            action_in = tf.layers.dense(self.actions, units=fc2_dims, activation='relu')

            state_actions = tf.add(batch2, action_in)
            state_actions = tf.nn.relu(state_actions)

            f3 = 0.003
            self.q = tf.layers.dense(state_actions, units=1, kernel_initializer=random_uniform(-f3, f3), bias_initializer=random_uniform(-f3, f3), kernel_regularizer=tf.keras.regularizers.l2(0.01))

            self.loss = tf.losses.mean_squared_error(self.q_target, self.q)

    def predict(self, inputs, actions):
        return self.sess.run(self.q, feed_dict={self.input: inputs, self.actions: actions})

    def train(self, inputs, actions, q_target):
        return self.sess.run(self.optimize, feed_dict={self.input: inputs, self.actions: actions, self.q_target: q_target})

    def get_action_gradients(self, inputs, actions):
        return self.sess.run(self.action_gradients, feed_dict={self.input: inputs, self.actions: actions})

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        self.saver.save(self.sess, self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.saver.restore(self.sess, self.checkpoint_file)


class Agent:

    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99, n_actions=2, buffer_size=1e6, batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.replay_buffer = ReplayBuffer(buffer_size)
        self.sess = tf.Session()

        self.actor = Actor(alpha, input_dims, n_actions, 'Actor', self.sess, env.action_space.high)
        self.critic = Critic(beta, input_dims, n_actions, 'Critic', self.sess)
        self.target_actor = Actor(alpha, input_dims, n_actions, 'TargetActor', self.sess, env.action_space.high)
        self.target_critic = Critic(beta, input_dims, n_actions, 'TargetCritic', self.sess)

        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.update_critic = [self.target_critic.params[i].assign(tf.multiply(self.critic.params[i], self.tau) + tf.multiply(self.target_critic.params[i], 1. - self.tau))
                              for i in range(len(self.target_critic.params))]

        self.update_actor = [self.target_actor.params[i].assign(tf.multiply(self.actor.params[i], self.tau) + tf.multiply(self.target_actor.params[i], 1. - self.tau))
                              for i in range(len(self.target_actor.params))]

        self.sess.run(tf.global_variables_initializer())

        self.update_network_parameters(first=True)

    def update_network_parameters(self, first=False):
        if first:
            old_tau = self.tau
            self.tau = 1.0
            self.target_critic.sess.run(self.update_critic)
            self.target_actor.sess.run(self.update_actor)
            self.tau = old_tau

        else:
            self.target_critic.sess.run(self.update_critic)
            self.target_actor.sess.run(self.update_actor)

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, done, next_state)

    def choose_action(self, state):
        state = state[np.newaxis, :]
        mu = self.actor.predict(state)
        noise = self.noise()
        mu_prime = mu + noise

        return mu_prime[0]

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, done, next_state = self.replay_buffer.sample(self.batch_size)

        critic_value_ = self.target_critic.predict(next_state, self.target_actor.predict(next_state))

        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma * critic_value_[j] * done[j])
        target = np.reshape(target, (self.batch_size, 1))

        _ = self.critic.train(state, action, target)

        a_outs = self.actor.predict(state)
        grads = self.critic.get_action_gradients(state, a_outs)
        self.actor.train(state, grads[0])

        self.update_network_parameters()

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()



