import gym
import numpy as np
from actor_critic import Agent
from utils import plot_learning_curve


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    agent = Agent(alpha=1e-5, n_actions=env.action_space.n)
    n_games = 1800

    filename = 'cartpole.png'
    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_model()
    
    for i in range(n_games):
        state = env.reset()
        done = False
        score = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            score += reward

            if not load_checkpoint:
                agent.learn(state, reward, next_state, done)
            
            state = next_state
        
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score

            if not load_checkpoint:
                agent.save_model()

        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)
        
    x = [i + 1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file)