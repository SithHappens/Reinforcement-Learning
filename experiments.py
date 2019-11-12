import gym
import numpy as np
from ddpg import Agent
from utils import plotLearning


env = gym.make('Pendulum-v0')

agent = Agent(alpha=0.0001, beta=0.001, input_dims=[3], tau=0.001, env=env, n_actions=1)

np.random.seed(0)
score_history = []

for episode in range(1000):

    state = env.reset()
    done = False
    score = 0

    while not done:
        action = agent.choose_action(state)

        next_state, reward, done, info = env.step(action)
        agent.remember(state, action, reward, next_state, int(done))

        agent.learn()

        score += reward
        state = next_state

    score_history.append(score)
    print('Episode {}, Score: {:.2f}, 100 game average: {:.2f}'.format(episode, score, np.mean(score_history[-100:])))

filename = 'pendulum.png'
plotLearning(score_history, filename, window=100)