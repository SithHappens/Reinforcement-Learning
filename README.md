# Dynamic Programming

Practical Limitations
- state space must be discrete and small enough for multiple iterations
- we rarely know the transition probabilities

## Value Iteration

- for MDPs with known transition probabilities and rewards
- has to iterate over all states multiple times

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

- doesn't need to iterate over full set of states anymore, but still too many.

For every state s in the MDP, perform the Bellman Update  
Q<sub>s,a</sub> = &sum;<sub>s'</sub> p<sub>a,s->s'</sub> (r<sub>s,a</sub> + &gamma; max<sub>a'</sub> Q<sub>s',a'</sub>)

> 1. Initialize all Q<sub>s,a</sub> to zero.
> 2. Interact with the environment and get (s, a, r, s')
> 3. Update Q<sub>s,a</sub> = (1-&alpha;) Q<sub>s,a</sub> + &alpha;(r + &gamma; max<sub>a'</sub> Q<sub>s',a'</sub>)
> 4. Repeat until convergence.


# Reinforcement Learning

**model-free**  

- no model of the environment or reward, current states are directly connected to actions (or values)
- easier to train in complex environments with rich observations


**model-based**  

- tries to predict what the next states / rewards will be
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

## Deep Q-Learning

> 1. Initialize Q<sub>s,a</sub> to some initial approximation.
> 2. Interact with the environment and get (s, a, r, s')
> 3. Calculate the loss  
L = (Q<sub>s,a</sub> - r)<sup>2</sup> &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp; if episode as ended,  
L = (Q<sub>s,a</sub> - (r + &gamma; max<sub>a'</sub> Q<sub>s',a'</sub>))<sup>2</sup> &emsp;&emsp;&emsp; otherwise
> 4. Update Q<sub>s,a</sub> using stochastic gradient descent by minimizing the loss with respect to the model parameters
> 5. Repeat until convergence.

- simple, but doesn't work very well
- initial actions are random, but the probability of success by random actions is small
- if the representation of Q is good, the experience will show the agent relevant data to train on, but if the approximation is bad, the agent can get stuck with bad actions for some states for ever  

**SGD Optimization**  
the fundamental requirement for SGD optimization is **indipendent and identically distributed** training samples, which is not fulfilled  

1. samples are not independent, because episodes  
2. distribution of training data (training policy) is not identical to samples of the optimal policy we want to learn. We don't want to learn to play randomly.  
    Solution: **Replay Buffer** instead of just using last experience
3. Bellman equation gives us Q(s,a) via Q(s',a'), called **bootstrapping**, but s and s' separates only one action:  
    When updating Q(s,a), we indirectly update Q(s',a') and other nearby states, making training unstable  
    Solution: **Target Network** for Q(s',a'), synced every N = 1k/10k/... steps

**Markov Property**
Inherited from the MDP formalism: Observation from the environment is all that we need to act optimally (i.e. observationsallow us to distinguish the states).

Single-frame pong for example is POMDP, because direction of the ball is not known. Also card games cause opponents hand is unknown.

Hack: Chain k consecutive observations together as the state, so dynamics can be deducted.

**Speed**  
Naive Implementation of the loss, which loops over every batch sample, is 2x slower than a parallel implementation. A single extra copy of the data batch could slow the same code 13x.  

## Actor-Critic

- Actor approximates policy &pi;
- Critic approximates V(s)
- Actor-Critic methods aren't really stable, but a stepping stone for more advanced methods

> **Temporal difference** &delta; = R<sub>t</sub> + &gamma;V(S<sub>t+1</sub>) - V(S<sub>t</sub>)  
>
> Critic Loss = &delta;<sup>2</sup>  
> Actor Loss = &delta; ln &pi;(A<sub>t</sub>|S<sub>t</sub>)

> **Algorithm**
> 
> Initialize Actor-Critic-Network  
> Repeat for large number of episodes:  
> * Reset environment, score, terminal flag  
> * While state is not terminal  
>   - Select action according to actor network  
>   - Take action, receive reward and new state  
>   - &delta; = R<sub>t</sub> + &gamma; V(S<sub>t+1</sub>)-V(S<sub>t</sub>)  
> * Plot scores over time for evidence of learning  

**Implementation Details**

- use one network, common lower layers (input), two outputs, no need to train two networks to understand the environment
- Softmax activation for actor and categorical distribution


