#general #markov #RL 
# Markov Property
- A state must include information about all aspects of past agent-environment interaction that makes a difference in the future
	$$
	P(s_{t+1}|s_t) = P(s_{t+1}|s_0,...,s_t)
	$$
# Markov Chain (MC)
- Mathematical system that experiences transition from one state to another according to probabilistic rules
- No matter how the process arrives at the the current state, possible future states are *fixed*
- Stochastic process that is "memory-less"
	- Stochastic: some values changing randomly over time
## Properties
- State $i$ has **period** $k \geq 1$ if any chain starting at and returning to $i$ must take a number of steps divisible by $k$ 
	- If $k=1$, then state is **aperiodic**
	- If $k > 1$, then state is **periodic**
	- If all states are aperiodic, then Markov chain is aperiodic
- A Markov chain is **irreducible** if there exists a chain of steps between any two states that has positive probability 
- A state is **recurrent (transient)** if Markov chain will eventually return to it
	- **Positive recurrent**: return within finite steps
	- **Null recurrent**: otherwise
- A state is **ergodic** if it is *positive recurrent and aperiodic*
	- A Markov chain is ergodic if all of its states are *ergodic*
# Markov Decision Process (MDP)
- Formalization (framework) for sequential decision making
- A 4-tuple $(\mathcal{S}, \mathcal{A}, p, \mathcal{R})$ where,
	- $\mathcal{S}$ is set of states called **state space**
	- $\mathcal{A}$ is set of actions called **action space**
	- $p(s,r|s,a) = Pr\{(S_{t}=s', R_t=r|S_{t-1}=s, A_{t-1}=a)\}$ is the **dynamics** of the MDP^dynamics
		- $p:\mathcal{S} \times \mathcal{R} \times \mathcal{S} \times \mathcal{A} \rightarrow [0,1]$ is an ordinary deterministic function of 4 arguments      
		- $p$ specifies the probability distribution for each choice of $s$ and $a$, that is,
		$$
		\sum_{s'\in \mathcal{S}} \sum_{r\in \mathcal{R}} p(s', r|s,a) = 1, \; \forall s \in \mathcal{S}, \mathcal{a} \in \mathcal{A}(s)
		$$
	- $\mathcal{R}$ is the reward function, which gives the reward received immediately after going from state $s$ to $s'$ due to action $a$
- MDP is an extension of Markov chain; different is addition of *actions and rewards*
	- If all actions are "no-op" and all rewards are zero, then MDP = MC
- From the 4-argument dynamics function $p$, one can compute anything else one might want to know about the environment,
## State-transition probabilities
- Denoted with slight abuse in notation, $p: \mathcal{S} \times \mathcal{S} \times \mathcal{A} \rightarrow [0,1]$
$$p(s'|s,a) = Pr\{S_t=s'|S_{t-1}=s, A_{t-1}=a\} = \sum_{r\in \mathcal{R}} p(s', r|s,a)$$
# Partially Observable Markov Decision Process (POMDP)
- Agent is no longer able to determine state it is in reliably
- Necessary to have *memory* of previous actions and observations to distinguish between states
- Important facet of POMDP approach is that there is no distinction between actions taken to change state and actions taken to gain info
	- Every action has both types of effects
	- Optimal performance involves “value of info” calculation; agent choose action based on amount of info they provide, amount of reward they produce, and how they change state of environment
## Belief state
- Agent maintains belief state; is a probability distribution over state space
- Summarize previous experience; State estimator updates belief state based on action and observation and current belief
- Policy is a function of belief state 
## Optimal policy 
- Is solution of continuous space “belief MDP”
- Very difficult to solve continuous space MDPs in general
## Reward function 
- Appears to be reward based on believing agent is in good state
- Works because state estimator is constructed from a correct observation and transition mole of the world
## Value functions for POMDPs
- Goal is to find approximate optimal value function; using value iteration to construct optimal t-step value function over belief space each iteration
- Policy trees (essentially equivalent to decision trees in decision theory to represent sequential decision policy)
	- Agent’s non-stationary t-step policy can be represented using a policy tree
	![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXednlZ5eE_gh6xFFrcstNaQ326bSwhPfpR4z8K3vqF1NsVt69H3075FcpfHjrrNhkor90NIOZbYjM1-vsVFBUN4VZ7IptMSORlf6F5-YoyRD0D7wVhFqPLef6n5Xyx7WNCE0fvznAYZIMGh6LXpLUzicAw?key=TO7E1UI3d4Js-AHQF9X4YA)