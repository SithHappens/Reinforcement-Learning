import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from networks import ActorCriticNetwork


class Agent:

    def __init__(self, alpha=0.0003, gamma=0.99, n_actions=2):
        self.gamma = gamma
        self.n_actions = n_actions
        self.actions = None
        self.action_space = [i for i in range(self.n_actions)]

        self.actor_critic = ActorCriticNetwork(n_actions=n_actions)
        self.actor_critic.compile(optimizer=Adam(learning_rate=alpha))

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])  # add batch dimension
        _, probs = self.actor_critic(state)

        action_probabilities =tfp.distributions.Categorical(probs=probs)
        action = action_probabilities.sample()
        self.action = action

        return action.numpy()[0]  # remove batch dimension

    def save_model(self):
        print('Saving model.')
        self.actor_critic.save_weights(self.actor_critic.checkpoint_file)

    def load_model(self):
        print('Loading model')
        self.actor_critic.load_weights(self.actor_critic.checkpoint_file)

    def learn(self, state, reward, next_state, done):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        next_state = tf.convert_to_tensor([next_state], dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)

        with tf.GradientTape() as tape:
            state_value, probs = self.actor_critic(state)
            next_state_value, _ = self.actor_critic(next_state)

            state_value = tf.squeeze(state_value)
            next_state_value = tf.squeeze(next_state_value)

            action_probs = tfp.distributions.Categorical(probs=probs)
            log_prob = action_probs.log_prob(self.action)

            delta = reward + self.gamma * next_state_value * (1 - int(done)) - state_value
            actor_loss = -log_prob * delta
            critic_loss = delta**2

            total_loss = actor_loss + critic_loss

        gradient = tape.gradient(total_loss, self.actor_critic.trainable_variables)
        self.actor_critic.optimizer.apply_gradients(zip(gradient, self.actor_critic.trainable_variables))
