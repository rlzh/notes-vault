#general #RL #markov 
- Computational approach to understanding and automating goal-oriented learning and decision making
- Uses the formal framework of [[Markov Decision Process]] to define the interaction between a learning agent and its environment in terms of states, actions, and rewards
- Goal is learning what to do (how to map states to actions) so as to maximize a numerical reward signal (function) over the long run; or *maximize the [[#Expected Return]]*
- Agent discovers which actions yield the most reward by trying them in the environment
- Inherent trade-off between exploration v.s. exploitation 
- Primary elements of RL include: [[#Reward Signal]], [[#Policy]], [[#Value Function]], and (optionally) a [[#Model of the Environment]]
# Prediction Problem
- The problem of estimating the [[#Value Function|value function]] $v_\pi$ (or action-value function $q_\pi$) for a given [[#Policy|policy]] $\pi$
# Control Problem
- The problem of finding an [[#Optimal Policies|optimal policy]]
# Agent-Environment Interface
![[agent-env-interface.png]]
- The agent and environment interact at each of a sequence of discrete (could also be continuous) time steps, $t=0,1,2,3...$
- At each time step $t$, the agent receives some representation of the environment's ***state***, $S_t \in \mathcal{S}$, and selects an ***action*** from the action set, $A_T \in \mathcal{A}(s)$
- On the next time step, the agent receives a numerical ***reward*** as a consequence of its action, $R_{t+1} \in \mathcal{R} \subset \mathbb{R}$ and arrives at the next state $S_{t+1}$
- This gives the *trajectory*, 
		$$
		S_0,A_0,R_1,S_1,A_1,R_2,...
		$$
## Model of the Environment
- Something that mimics the behaviour of the environment and allows inference to be made about how the environment will behave
	- e.g., given a state and action, the model can predict the resultant next state and next reward
- Models are used for *planning* (i.e., deciding on a course of action by considering possible future situations before they are experienced)
## Model-based RL
- Require a *model of the environment*
- Rely on *planning* as their primary component
	- e.g., dynamic programming, heuristic search

## Model-free RL
- Methods that can be used without a *model of the environment*
- Rely on *learning* as their primary component
	- e.g., Monte Carlo, temporal difference


# Expected Return
## Episodic Tasks
- The agent-environment interaction naturally can be divided into **episodes**
- The episode ends in a special state called the **terminal state**
- $T$ is the final time step
- If the sequence of rewards received after each time step $t$ is denoted $R_{t+1}, R_{t+2}, ...$, the expected return $G_t$ is defined as,
$$G_t = R_{t+1} + R_{t+2} + ... + R_T$$
## Continuing Tasks
- the agent-environment interaction does not break naturally into identifiable episodes; but goes on continually without limit
- The final time step $T = \infty$     
- Require the concept of **discounting**, where immediate rewards may be more impactful than future rewards
- The **expected *discounted return*** is defined as,
	$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... = \sum^{\infty}_{k=0} \gamma^k R_{t+k+1}$$ where, 
- $\gamma$ is a parameter, $0 \leq \gamma \leq 1$ called the **discount rate**
	- If $\gamma = 0$, the agent is *"myopic"* in being concerned with only maximizing immediate rewards
	- As $\gamma$ approaches 1, the agent becomes more farsighted
- Returns in successive time steps are related to each other in the following fashion,	$$
		\begin{align}
		G_t &= R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... \\
		    &= R_{t+1} + \gamma (R_{t+2} + \gamma R_{t+3} + ...) \\
		    &= R_{t+1} + \gamma (G_{t+1}) 
		
		\end{align}
		$$
## Unified Notation
- Notation for expected return for episodic and continuing tasks can be unified with the introduction of a special **absorbing state**
	- The absorbing state generates only rewards of zero
		![[absorbing-state.png]]
- The general notation can be written as,
	$$G_t = \sum^{T}_{k=t+1} \gamma^{k-t-1} R_k$$ where $T=\infty$ or $\gamma=1$ (but not both)
# Reward Signal
- Defines the goal of a RL problem
- Each time step, the environment sends to the agent a single number called the *reward*
- Defines what are the *good and bad* events for the agent
	- Analogous to "pleasure" or "pain"
# Policy
- Algorithm an agent uses to determine its action in a given state
- Maps from states to probabilities of selecting each possible action
- If the agent is following policy $\pi$ at time $t$, then $\pi(a|s)$ is the probability that $A_t = a$ if $S_t=s$
	- The "$|$" in the middle of $\pi(a|s)$ defines a probability distribution over $a \in \mathcal{A}(s)$ for each $s \in \mathcal{S}$ 
## Optimal Policies
- We can precisely define an *optimal* policy using [[#Value Function|value functions]]
- A policy $\pi$ is defined to be better than or equal to a policy $\pi'$ if its [[#Expected Return|expected return]] is greater than or equal to that of $\pi'$ for all states
- $\pi \geq \pi'$ if and only if $v_\pi(s) \geq v_{\pi'}(s)$ for all $s \in \mathcal{S}$
- **There is always at leas one policy that is better than or equal to all other policies**; this is called the *optimal policy* denoted by $\pi_{*}$ 
## On-policy methods
- Evaluate or improve the policy that is used to make decisions
## Off-policy methods
- Evaluate or improve a policy different from that used to generate the data
# Value Function
- Specifies what is good in the long run (long-term desirability of a state)
- The ***value*** of a state is the total amount of reward an agent can expect to accumulate over the future, starting from that state
## State-value function
- The value function of a state $s$ under policy $\pi$ is denoted $v_\pi(s)$
	- It is the [[#Expected Return]] of starting in $s$ and following $\pi$ thereafter
	$$
		v_\pi(s) = \mathbb{E}_\pi[G_t|S_t = s] = \mathbb{E}_\pi[\sum^{\infty}_{k=0}\gamma^k R_{t+k+1} | S_t = s], \; \forall s \in \mathcal{S}
	$$
	- The value of the terminal state, if any, is always zero
### Optimal State-value function
- Shared by the [[#Optimal Policies]] and is denoted by $v_*$ and defined as,
	$$v_*(s) = \max_{\pi} v_\pi(s), \; \forall s \in \mathcal{S}$$
## Action-value function
- The value of taking action $a$ in state $s$ under a policy $\pi$ is denoted $q_\pi(s,a)$
	- It is the [[#Expected Return]] starting from $s$, taking action $a$, and following $\pi$ thereafter
	$$
	q_\pi(s,a) = \mathbb{E}_\pi[G_t | S_t=s, A_t=a] = \mathbb{E}_\pi[{\sum^\infty_{k=0} \gamma^{k} R_{t+k+1}|S_t=s, A_t=a}]
	$$
### Optimal Action-value function
- Shared by the [[#Optimal Policies|optimal policies]] and is denoted by $q_*$ and is defined as,
	$$q_*(s,a) = \max_{\pi} q_\pi(s,a), \; \forall s \in \mathcal{S}, a \in \mathcal{A}(s)$$
- We can also write $q_*$ in terms of $v_*$ as follows,
	$$q_*(s,a) = \mathbb{E}[R_{t+1}+\gamma v_*(S_{t+1})|S_t = s, A_t=a]$$

## Expected Reward for State-Action pairs
- $r: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \rightarrow \mathbb{R}$ 
$$r(s,a) = \mathbb{E}[R_t|S_{t-1}=s, A_{t-1} = a] = \sum_{r\in \mathcal{R}}r\sum_{s' \in \mathcal{S}}p(s',r|s,a)$$
## Bellman Equation
- For any policy $\pi$ and any state $s$, the following consistency condition holds between the value of $s$ and the value of its possible successor states,
	$$
	\begin{align}
	v_\pi(s) &= \mathbb{E}_\pi[G_t|S_t = s] \\
	&= \mathbb{E}_\pi[R_{t+1}+\gamma G_{t+1} | S_t = s] \\
	&= \sum_{a} \pi(a|s) \sum_{s'} \sum_{r} p(s', r|s, a) [r + \gamma \mathbb{E}_\pi[G_{t+1}|S_{t+1}=s']] \\
	&= \sum_{a} \pi(a|s) \sum_{s'} \sum_{r} p(s',r|s,a)[r + \gamma v_\pi(s')], \; \forall s \in \mathcal{S}

	\end{align}
	$$
- This expresses the relationship between the value of a state and the values of its successor states
- It averages over all possibilities, weighting each by its probability of occurring 
	- It states that the value of the start state must be equal to the (discounted) value of the expected next state, plus the rewards expected along the way
	![[backup-bellman-state-value-function.png|150]]
	![[backup-bellman-action-value-function.png|150]]
## Bellman Optimality Equation
- The [[#Optimal State-value function]] $v_*$ must also satisfy the self-consistency condition given by the [[#Bellman Equation]]; thus allowing $v_*$ to be expressed in a special form without reference to any specific policy as follows
	$$
	\begin{align}
	v_*(s) &= \max_{a \in \mathcal{A}(s)} q_{\pi_{*}}(s,a) \\
	&= \max_{a} \mathbb{E}_{\pi_*}[G_t | S_t = s, A_t = a] \\
	&= \max_{a} \mathbb{E}_{\pi_*}[R_{t+1} + \gamma G_{t+1} | S_t = s, A_t = a] \\
	&= \max_{a} \mathbb{E}_{\pi_*}[R_{t+1} + \gamma v_{*}(S_{t+1}) | S_t = s, A_t = a] \\
	&= \max_{a} \sum_{s',r} p(s', r| s,a) [r+ \gamma v_*(s')]
	\end{align}
	$$
- It expresses the fact that the value of a state under an [[#Optimal Policies|optimal policy]] must equal the expected return for the best action from that state
- The Bellman optimality equation for $q_*$ is,
	$$
	\begin{align}
	q_*(s,a) &= \mathbb{E}[R_{t+1} + \gamma \max_{a'}q_{*}(S_{t+1}, a')|S_t=s, A_t = a] \\
	&= \sum_{s',r}p(s',r|s,a)[r+\gamma\max_{a'}q_*(s',a')]
	\end{align}
	$$
	![[backup-bellman-opt.png|400]]
- The Bellman optimality equation is actually a system of equations, one for each state
	- If there are $n$ states, then there are $n$ equations in $n$ unknowns
- Once $v_*$ is known, it is easy to to determine an optimal policy
	- For each state $s$, there will be one (or more) actions which obtains the maximum return based on the Bellman optimality equation
	- Any policy that assigns non-zero probability only to these actions is an optimal policy
- Explicitly solving Bellman optimality equation is rarely directly useful; akin to exhaustive search!
	- Requires that we know the dynamics of the system
	- Requires we have enough computational resources to complete the computation
	- Requires the [[Markov Decision Process#Markov Property|Markov Property]]
# Action Value
- The mean reward when an action is selected in a given state
- Denoted as $q_*(a) = \mathbb{E}[R_t|A_t=a]$, which is generally *unknown* (this what we solve for)
- The action value is estimated naturally by,
$$
Q_t(a) = \frac{\text{sum of rewards when} \; a \; \text{taken prior to} \; t}{\text{number of times} \; a \; \text{taken prior to } \; t} = \frac{\sum^{t-1}_{i=1}R_i\cdot \mathbb{1}_{A_i=a}}{\sum^{t-1}_{i=1}\mathbb{1}_{A_i=a}}
$$
where,
- $\mathbb{1}_{predicate}$ is the random variable that is 1 if $predicate$ is true and 0 if not.
- Goal is to make $Q_t(a)$ be close to $q_*(a)$
# Tabular Solution Methods

## Multi-armed Bandit (k-armed bandit)
- **Problem**: agent repeatedly face with a choice among $k$ options (or actions)
	- After each choice, a numerical reward is chosen from a *stationary* probability distribution that depends on the action you selected
	- Objective is to maximize the expected total reward over some time period (e.g., over 1000 action selections or time steps)
### Optimistic Initial Value
- Over estimate action values to encourage exploration at the early stages of learning
- Only works well for *stationary* learning problems

## Dynamic Programming (DP)

### Policy Evaluation
- Deals with [[#Prediction Problem]]
- Recall for all $s\in \mathcal{S}$ 
$$
\begin{align}
V_\pi(s) &= \mathbb{E}_\pi[G_t|S_t=s] \\
	&= \mathbb{E}_\pi[R_{t+1} + \gamma v_\pi(S_{t+1})| S_t = s] \\
	&= \sum_{a}\pi(a|s)\sum_{s',r}p(s',r|s,a)[r+\gamma v_\pi(s')]
\end{align}
$$
- If the environment's dynamics are completely known, then the above is a system of $|\mathcal{S}|$ simultaneous linear equations in $|\mathcal{S}|$ unknowns (the $v_\pi(s)$ values)
- Under ***iterative policy evaluation***, we initialize $v$ arbitrarily and iteratively approximate $v_\pi$ using [[#Bellman Equation]] as update rule,
	$$
	\begin{align}
	v_{k+1}(s) &= \mathbb{E}_\pi[R_{t+1} + \gamma v_k(S_{t+1})|S_t = s] \\
	&= \sum_{a} \pi(a|s) \sum_{s',r}p(s',r|s,a) [ r+ \gamma v_k(s')]
	\end{align} 
	$$
	- As $k \rightarrow \infty$, it can be shown that $v_k$ in general converges to $v_\pi$  
- The update from $v_k$ to  $v_{k+1}$ is called ***expected update***
	- Each iteration of iterative policy evaluation updates the value of *every* state once to produce the new approximate value function $v_{k+1}$ 
	- Expected updates can be done "in place"; which converges faster generally
	![[iterative-policy-eval-algo.png]]
### Policy Improvement
- Motivation for computing value function ([[#Policy Evaluation]]) is to find better policies
- For some state $s$, one might want to know whether policy would be improved to choose an action $a \neq \pi(s)$  
	- To answer this question, consider selecting $a \neq \pi(s)$ in $s$ and then following the policy $\pi$ thereafter; the value of this is,
	$$
	\begin{align}
	q_\pi(s,a) &= \mathbb{E}[R_{t+1} + \gamma v_\pi(S_{t+1})|S_t=s, A_t=a ] \\
		&= \sum_{s',r} p(s',r|s,a)[r+ \gamma v_\pi(s')]
	\end{align}
	$$
	- The action $a$ is better if $q_\pi(s,a) \ge v_\pi(s)$ 
- **General policy improvement theorem***: let $\pi$ and $\pi'$ be any pair of deterministic policies such that, for all $s \in \mathcal{S}$,
		$$q_\pi(s, \pi'(s)) \geq v_\pi(s)$$ ^ec947c
	- Then policy $\pi'$ must be as good as, or better than $\pi$ (i.e., $v_{\pi'}(s) \geq v_\pi(s)$ for all states)
	- This theorem can be proven by expanding $q_\pi(s, \pi'(s))$ until we eventually arrive at $v_{\pi'}(s)$
- **Policy improvement** works by considering the *greedy* policy $\pi'$ given by
	$$ 
	\begin{aligned}
	\pi'(s)&= arg\max_{a}q_\pi(s,a) \\
		 &= arg\max_{a}\sum_{s',r}p(s',r|s,a)[r+ \gamma v_\pi(s')]
	\end{aligned}
	$$
	- This policy takes the best action in the short term according $v_\pi$ and meets the condition of general policy improvement theorem above ([[#^ec947c]])  
	- If some new greedy policy $\pi'$ is as good as but not better than the old policy $\pi$, then $v_\pi' = v_\pi$ for all states and,
		$$v_\pi'(s) = \max_a \sum_{s',a} p(s',r|s,a)[r+\gamma v_{\pi'}(s')]$$
	- The above is the same as the [[#Bellman Optimality Equation]], therefore $v_{\pi'}$ must be $v_*$ and $\pi$, $\pi'$ must be [[#Optimal Policies|optimal policies]]
	- For stochastic case, any apportioning scheme is allowed for the policy as long as all sub-maximal actions are given zero probability	
### Policy Iteration
- Once a policy $\pi$ has been improved using $v_\pi$ to give an improved policy $\pi'$, we can compute $v_{pi'}$ and improve it again to give an even better $\pi''$...
	$$\pi_0 \xrightarrow{E} v_{\pi_{0}} \xrightarrow{I} \pi_1 \xrightarrow{E} v_{\pi_1} \xrightarrow{I}... \xrightarrow{I} \pi_* \xrightarrow{E} v_*$$
- The above way of finding an [[#Optimal Policies|optimal policy]] is called **policy iteration***
	![[policy-iteration-algo.png]]
### Value Iteration
- Avoids drawback of [[#Policy Evaluation|policy evaluation]] which may be a expensive computation that requires multiple sweeps through the state set
- Main idea is to truncate policy evaluation step of policy iteration and combines [[#Policy Improvement|policy improvement]] and truncated policy evaluation steps
	$$v_{k+1}(s) = \max_{a}\sum_{s',r}p(s',r|s,a)[r+\gamma v_k(s')]$$
	- Value iteration is obtained by using [[#Bellman Optimality Equation]] as update rule
	- It is identical to [[#Policy Evaluation|policy evaluation]] update rule except it requires the *maximum* to be taken over all actions
- Effectively combines in each sweep, one sweep of policy evaluation and one sweep of policy improvement
![[value-iteration-algo.png]]
### Generalized Policy Iteration
- [[#Policy Iteration]] contains two simultaneous interacting processes
	- One making the value function consistent with the policy ([[#Policy Evaluation|policy evaluation]])
	- One making the policy *greedy* with respect to the current value function ([[#Policy Improvement|policy improvement]])
	- The two processes alternate 
- [[#Value Iteration]] only a single iteration of policy evaluation is performed between each policy improvement
- ***Generalized Policy Iteration (GPI)*** to refers to the idea of letting policy evaluation and policy improvement processes interact
	- Almost all RL algorithms are described as GPI; all have identifiable policies and value functions
	- Policy is improved with respect to value function and value function driven toward the value function for the policy
	 ![[gpi-relationship.png|150]]    ![[GPI-flow.png|300]]


## Monte Carlo (MC)
- Do not assume complete knowledge of the environment
- MC methods require only *experience* (i.e., sample sequences of states, actions, and rewards) from actual or simulated interaction with an environment
	- [[#Dynamic Programming (DP)]] methods require complete probability distributions of all possible transitions
- MC methods are solve RL problems based on *averaging sample returns*
- Only works for episodic tasks; i.e., experience is divided into episodes and all episodes eventually terminate (no matter the actions selected)
- Value estimates and policies are changed *only on the completion of episodes*
	- MC methods are incremental in episode-by-episode sense
	- Learn value functions from sample returns with the [[Markov Decision Process]]
	- The value functions and corresponding policies still interact to obtain optimality using [[#Generalized Policy Iteration]]
### MC Prediction
- Deals with the [[#Prediction Problem|prediction problem]]
### MC Estimation of Action Values

### MC Control

### Off-policy Prediction

### Off-policy Control

### Monte-Carlo Tree Search

### UCT Algorithm
- Upper Confidence bounds for Trees
- Algorithm deals with a flaw of Monte-Carlo Tre

## Temporal-Difference (TD)
- Combination of [[#Monte Carlo (MC)]] and [[#Dynamic Programming (DP)|dynamic programming (DP) ideas]]
	- Like Monte Carlo, TD learns directly from raw experience *without model of the environment's dynamics*
	- Like DP, TD updates estimates based on other learned estimates, without waiting for final outcome (bootstrap)
- Central and novel to RL
### TD Prediction
- [[#Monte Carlo (MC)|Monte Carlo (MC) methods]] wait until the return following the visit is known, then use that return as a target for $V(S_t)$
	- Simple every-state MC method suitable for non-stationary environment, called *constant*-$\alpha$ MC
	$$V(S_t) \leftarrow V(S_t) + \alpha[G_t-V(S_t)]$$
- Whereas MC methods must wait until end of episode to determine increment to $V(S_t)$, TD methods only need to wait until next time step $t+1$ to make a useful update based on observed reward $R_{t+1}$,
	$$V(S_t) \leftarrow V(S_t) + \alpha[R_{t+1}+\gamma V(S_{t+1})-V(S_t)]$$
- The above is called ***TD(0) or one-step TD***
![[tabular-TD(0).png]]
- TD(0) is a ***bootstrapping*** method (like DP)
$$
\begin{align}
v_\pi(s) &= \mathbb{E}[G_t | S_t = s] \\
 &= \mathbb{E}[R_{t+1} + \gamma G_{t+1} | S_t = s] \\
 &= \mathbb{E}[R_{t+1} + \gamma v_\pi(S_{t+1}) | S_t = s]
\end{align}
$$
	- MC uses an estimate of $G_t$ as a target (sample return is used in place of real expected return)
	- DP uses an estimate of $R_{t+1} + \gamma v_\pi(S_{t+1})$ as target (because $v_\pi(S_{t+1})$ is not known, the current estimate $V(S_{t+1})$ is used instead)
	- TD target is an estimate for both reasons; it samples the expected values in  $R_{t+1} + \gamma v_\pi(S_{t+1})$ and it uses the current value of $V$ instead of the true $v_\pi$
- TD combines sampling of MC and bootstrap of DP
- TD (and MC) updates are referred to as ***sample updates***; because they involve looking ahead to a sample successor state
- ***TD error*** is a measurement of difference between estimated value of $S_t$ and the better estimate $R_{t+1} + \gamma V(S_{t+1})$ 
	$$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$
	- MC error can be written as sum of TD errors
		$$G_t - V(S_t) = \sum^{T-1}_{k=t}\gamma^{k-t}\delta_k$$
### Advantages of TD Prediction
- TD methods update their estimates based in part on other estimates (learn a guess from a guess - i.e., ***bootstrap***)
- Implemented in an online, incremental fashion
- Do not require a model of the environment
### Optimality of TD(0)
- ***Batch updating***: when updates to value estimates are made only after processing each complete *batch* of training data
	- e.g., when there is only a finite amount of experience (e.g., 100 time steps)
- Under batching update, TD(0) converges deterministically to a single answer independent of step-size parameter $\alpha$ (as long as $\alpha$ is chosen sufficiently small)
	- Batch MC methods always find the estimates that minimize mean-squared error on the training set
	- Batch TD(0) always finds the estimates that would be exactly correct for *maximum-likelihood* model of the Markov process
- ***Maximum-likelihood estimate*** of a parameter is the parameter value whose probability of generating the data is greatest
- ***Certainty-equivalence estimate*** assumes that the estimate of the underlying process was known with certainty rather than being approximated
	- Batch TD(0) converges to the certainty-equivalence estimate
	- It is (in some sense) an optimal solution, but it is never feasible to compute it directly without using TD
### Sarsa
- [[#On-policy methods|On-policy]] TD [[#Control Problem|control]] method
- First step is to learn [[#Action Value]] function; estimate $q_\pi(s,a)$ for current behaviour policy $\pi$ and for all states $s$ and actions $a$ 
- This is similar to the idea of [[#TD Prediction]], except applied to state-action pairs
	$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha[R_{t+1} + \gamma Q(S_{t+1},A_{t+1})-Q(S_t, A_t)]$$
	![[sarsa-control.png]]
### Q-learning
- [[#Off-policy methods|Off-policy]] TD [[#Control Problem|control]] algorithm defined by the rule,
	$$Q(S_t,A_t) \leftarrow Q(S_t, A_t) + \alpha[R_{t+1}+\gamma \max_{a} Q(S_{t+1},a) - Q(S_t, A_t)]$$
- The learned action-value function $Q$ directly approximates the optimal action-value function $q_*$, independent of the policy being followed 
![[q-learning-algo.png]]
### Expected Sarsa
- Learning algorithm similar to [[#Q-learning]] except it uses the *expected value* for next state-action pair (instead of the maximum)
- The algorithm follows the rule,
	$$
	\begin{align}
	Q(S_t, A_t) &\leftarrow Q(S_t, A_t) + \alpha[R_{t+1} + \gamma \mathbb{E}[Q(S_{t+1}, A_{t+1})| S_{t+1}] - Q(S_t, A_t)] \\
	&\leftarrow Q(S_t, A_t) + \alpha[R_{t+1} + \gamma \sum_{a} \pi(a|S_{t+1})Q(S_{t+1, a}) - Q(S_t, A_t)]
	\end{align}
	$$
- Eliminates variance due to random selection of $A_{t+1}$ compared to [[#Sarsa]]
# N-step Bootstrapping
todo...

