import argparse


if __name__ == '__main__':
    # 'BipedalWalker-v2', 'BipedalWalkerHardcore-v2', 'CarRacing-v0', 'HandReach-v0', 'MountainCarContinuous-v0'
    env_list = ['Taxi-v3', 'CartPole-v1', 'MountainCar-v0', 'Ant-v2', 'LunarLander-v2', 'LunarLanderContinuous-v2']
    alg_list = ['cross_entropy']

    parser = argparse.ArgumentParser(description='Train Reinforcement Learning Agents in different environments. You need to specify the agent and the environment.')
    parser.add_argument('--render', '-r', action='store_true', help='Render during the learning process.')  # type=int, dest=var_name
    parser.add_argument('agent', choices=alg_list, )   #, required=True)
    parser.add_argument('env', choices=env_list)    #, required=True)

    args = parser.parse_args()

    # TODO call agents and actually train. How can the different neural networks be handeled?