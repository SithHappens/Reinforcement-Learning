# Reinforcement Learning Arena

- **model-free**
	- no model of the environment or reward, current states are directly connected to actions (or values).
	- easier to train in complex environments with rich observations
	

- **model-based**
	- tries to predict what the next states / rewards will be.
	- for deterministic environments
	
- **policy-based**
	- directly approximate the policy (probability distribution over available actions)
	- state s &rarr; neural network &rarr; policy &pi;(a|s)
	
- **value-based**
	- estimate the value of every possible action

- **on-policy**

- **off-policy**
	- can learn on historical data (replay buffer)

## Practical Cross-Entropy

- model-free, policy-based, on-policy
- core idea: throw away bad episodes and train on the better ones.
- robust, converges quickly

> 1. 	Play N number of episodes using the current model and environment.
> 2.	Calculate the total reward per episode, decide on a reward threshold (e.g. 50% or 70% of all rewards).
> 3. 	Throw away all episodes with rewards below the threshold.
> 4.	Train on remaining episodes using states as input and issued actions as desired output.
> 5.	Repeat until convergence.

