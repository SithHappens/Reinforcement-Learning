# Dynamic Programming

Practical Limitations
- state space must be discrete and small enough for multiple iterations
- we rarely know the transition probabilities

## Value Iteration

- for MDPs with known transition probabilities and rewards

> 1. Initialize values of all states V<sub>i</sub>, usually zero.
> 2. For every state s in the MDP, perform the Bellman Update  
V<sub>s</sub> = max<sub>a</sub> &sum; <sub>s'</sub> p<sub>a,s->s'</sub> (r<sub>s,a</sub> + &gamma; V<sub>s'</sub>)
> 3. Repeat step 2 until convergence.

Implementation
- reward table  
    dict { (s, a, s') : immediate reward }
- transitions table to estimate transition probabilities  
    dict { (s,a) : dict { s' : count } }
- value table  
    dict { s : V<sub>s</sub> }

## Q-Learning

> 1. Initialize all Q<sub>s,a</sub> to zero.
> 2. For every state s in the MDP, perform the Bellman Update  
Q<sub>s,a</sub> = &sum;<sub>s'</sub> p<sub>a,s->s'</sub> (r<sub>s,a</sub> + &gamma; max<sub>a'</sub> Q<sub>s',a'</sub>)
> 3. Repeat step 2 until convergence.


# Reinforcement Learning

**model-free**  
- no model of the environment or reward, current states are directly connected to actions (or values).
- easier to train in complex environments with rich observations


**model-based**  
- tries to predict what the next states / rewards will be.
- for deterministic environments

**policy-based**  
- directly approximate the policy (probability distribution over available actions)
- state s &rarr; neural network &rarr; policy &pi;(a|s)

**value-based**  
- estimate the value of every possible action

**on-policy**  

**off-policy**  
- can learn on historical data (replay buffer)

## Loss Functions

**MSELoss**  
Standard for regression problems

**BCELoss and BCEWithLogits**  
Binary Cross Entropy for binary classification. First one expects single probability value (e.g. sigmoid layer output), second assumes raw scores and applies sigmoid itself (more efficient).

**CrossEntropyLoss and NLLLoss**  
'Maximum Likelihood' criteria, for multi-class classification. First expects raw scores and applies logSoftmax, second expects log probabilities as input.

## Practical Cross-Entropy

- model-free, policy-based, on-policy
- core idea: throw away bad episodes and train on the better ones.
- simple, robust, converges quickly
- can only learn from complete episodes
- for simple environments with short, finite episodes and frequent rewards
- reward for episodes needs enough variability to seperate good episodes from bad ones

> 1. Play N number of episodes using the current model and environment.
> 2. Calculate the total reward per episode, decide on a reward threshold (e.g. 50% or 70% of all rewards).
> 3. Throw away all episodes with rewards below the threshold.
> 4. Train on remaining episodes using states as input and issued actions as desired output.
> 5. Repeat until convergence.

- works best if the agent gets a reward for every step, so different episodes with different lengths give a pretty normal distribution of the episodes rewards, so less successful ones (short ones) can be rejected.
- does not work if reward is given for reaching the goal only, because this reward says nothing about how good/efficient the episodes was. Failed episodes dominate in the beginning, training on bad percentile selection leads to training failure.

