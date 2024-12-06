#RL #paper #bayesian 
**author:** Guez
**conference/journal:** NIPS
**year**: 2012
**file:** [[2012-NIPS-efficient-bayes-adaptive-reinforcement-learning-using-sample-based-search-Guez.pdf]]
# Summary
- Present Monte-Carlo tree search (MCTS) to solve BAMDP [[Reinforcement Learning#Control Problem|problem]] 
- Introduce lazy sampling scheme that reduces cost of sampling
- Proposed approach does not update posterior belief state during the course of each simulation
	- Avoids repeated applications of [[Bayes' Theorem|Bayes rule]]
	- Well-suited for planning in domains with ***richly structured prior knowledge*** ([[2012-NIPS-efficient-bayes-adaptive-reinforcement-learning-using-sample-based-search-Guez.pdf#page=2&selection=4,0,4,59|ref]])
- Compared algorithm to other Bayesian RL and non-Bayesian approaches to showcase effectiveness 
## Motivation
- Deal with ***unknown dynamics*** of [[Markov Decision Process]]
- Provide solution that is scalable (relatively) for solving Bayes-Adaptive MDP (BAMDP)
# Approach

## Bayesian RL
- Generic Bayesian formulation of optimal decision-making in an *unknown MDP*
- A [[Markov Decision Process]] is described as a tuple $M = \langle S, A, \mathcal{P}, \mathcal{R}, \gamma\rangle$ 
	- When all components of the tuple are known, standard MDP planning algorithms can be used to estimate the [[Reinforcement Learning#Optimal State-value function|optimal value function]] and policy off-line
	- But, in general, the ***dynamics are unknown***
## BAMDP
- Assume $\mathcal{P}$ (the state transition probability) is a [[Statistics#Latent Variable|latent variable]] distributed according to $P(\mathcal{P})$ 
- After observing a ***history of actions*** and states from the MDP
	- $h_t = s_1,a_1,s_2,a_2,...,a_{t-1},s_t$ 
	- the posterior belief on $\mathcal{P}$ is updated using [[Bayes' Theorem]] 
- Uncertainty about the dynamics of the model can be transformed into uncertainty about the current state inside an **augmented state space $S^+=S\times\mathcal{H}$**
	- $S$ is the original state space
	- $\mathcal{H}$ is the set of possible histories
- The dynamics associated with the augmented state space is described by
	- Augmented state transition probability $$\mathcal{P}^+(\langle s,h\rangle, a, \langle s', h' \rangle) = \mathbb{1}[h'=has]\int_{\mathcal{P}} \mathcal{P}(s,a,s')P(\mathcal{P}|h)d\mathcal{P}$$
	- Augmented reward function
		$$
		\mathcal{R}^+(\langle s,h \rangle, a) = R(s,a)
		$$
- Together, **the tuple $M^+ = \langle S^+, A, \mathcal{P}^+, \mathcal{R}^+, \gamma \rangle$ forms the Bayes-Adaptive MDP (BAMDP)** for the MDP problem $M$
	- Since the dynamics of the BAMDP are known, in can be solved to obtain the optimal action-value function (and the optimal policy from this)
		$$
		Q^*(\langle s_t, h_t \rangle, a) = \max_\pi \mathbb{E}_\pi [\sum^\infty_{t'=t} \gamma^{t'-t} r_{t'}|a_t =a]
		$$
- Optimal actions in the BAMDP $M^+$ are executed *greedily* in the MDP $M$
	- This constitutes the best action for a Bayesian agent based on its prior belief over $\mathcal{P}$
- The expected performance of the BAMDP policy $\pi^{*+}$ in the MDP model is bounded by the optimal policy $\pi^*$ in the original MDP model
	- Equality occurs when the prior only has support for the true model (dynamics)
## BAMCP 
- Bayes-adaptive Monte-Carlo Planning (BAMCP) performs forward-search in the space of possible future histories of the [[#BAMDP]] using tailored Monte-Carlo tree search
- Employ UCT algorithm to allocate search effort to *promising* branches of the state-action tree
	- Use sample-based rollouts to provide value estimates at each node
	- Naively applying UCT to BAMDP (denoted BA-UCT) requires generation of samples from $\mathcal{P}^+$ at *every node*
	- This operation requires integration over all possible transition models, which is ***very expensive*** 
- Root node of the search tree at a decision point represents the current state of the BAMDP
- ***Tree*** is composed of 
	- **state nodes**, representing belief states $\langle s, h \rangle$ 
	- **actions nodes**, representing the actions from their parent state nodes
		- A value $Q(\langle s,h \rangle, a)$, initialized to 0, is maintained for each action node
	- The **visit counts** for state nodes $N(\langle s, h \rangle)$ and action nodes $N(\langle, s,h \rangle, a)$ are initialized to 0 and updated through *search*
- Each simulation traverses the tree without backtracking by following the UCT policy at state nodes defined by
		$$ arg\max_a Q(\langle s,h \rangle, a) + c \sqrt{\frac{log(N(\langle s,h\rangle))}{N(\langle s,h \rangle, a)}}$$
	- where $c$ is an exploration constant (needs to be set *appropriately*)
![[bamcp-algo.png]]
### Root Sampling
- **Main idea:** avoid the above *expensive* operation by only sampling a single transition model $\mathcal{P}^i$ from the posterior at the root of the search tree at the start of each simulation $i$
	- Use $\mathcal{P}^i$  to generate all the necessary samples during the current simulation
	- This is denoted as *Root Sampling*
- Given an action from the UCT policy (defined above), the transition probability $\mathcal{P^i}$ corresponding to the current simulation $i$ is used to sample the next state
	- i.e., at action node $(\langle s,h \rangle, a)$ , the next state $s'$ is sampled from $\mathcal{P}^i(s,a,\cdot)$ 
	- the new state node is then set to $\langle s', has' \rangle$ 
- When **a simulation reaches a *leaf node***, the tree is expanded by attaching a new state node with its connected action nodes
	- A **rollout policy** $\pi_{ro}$ is used to control the MDP defined by the current $\mathcal{P}^i$ to some fixed depth (determined using discount factor $\gamma$) 
	- The rollout provides an estimate of the value $Q(\langle s,h \rangle, a)$ from the leaf action node
	- This estimate is used to update the value of all action nodes traversed during the simulation
	- If $R$ is the sampled [[Reinforcement Learning#Expected Return|discounted return]] obtained from traversing an action node during the simulation, then we update the value of the action node to the mean of the sampled returns obtained from the action node over simulations, given by
		$$
		Q(\langle s,h \rangle, a) \leftarrow Q(\langle s,h \rangle, a) + \frac{R-Q(\langle s,h \rangle, a)}{N(\langle s,h \rangle, a)} 
		$$
- The tree policy treats the forward search as a meta-exploration problem
	- Prefer to exploit regions of tree that currently appear to be better than others (using UCT), while also continuing to explore unknown or lesser known parts
	- All parts of tree are eventually visited *infinitely many times*, which ensures the algorithm converges to *Bayes-optimal policy*
- **Note:** the history of transitions $h$ is generally not very compact sufficient statistic of belief in MDPs
	- Can be replaced with unordered transition counts $\psi$ to reduce the number of states (and complexity) in the BAMDP considerably 
### Lazy Sampling
- **Main idea**: even [[#Root Sampling]] can be very computationally costly
	- Propose sampling $\mathcal{P}$ *lazily* 
	- i.e., creating only the particular transition probabilities that are *required* as the simulation traverses the tree (and also during rollout)
- Consider $\mathcal{P}(s,a,\cdot)$ to be parameterized by a latent variable $\theta_{s,a}$ for each state and action pair
	- These may be depended on each other, and an additional set of latent variables $\phi$ 
	- The posterior over $\mathcal{P}$ can be written as 
		$$
		P(\Theta|h) = \int_\phi P(\Theta|\phi, h)P(\phi|h) 
		$$
	- where $\Theta = \{\theta_{s,a}|s \in S, a \in A \}$
	- Define $\Theta_t = \{\theta_{s_1, a_1}, ..., \theta_{s_t, a_t}\}$ as the (random) set of $\theta$ parameters required during the course of a BAMCP simulation that starts at time $1$ and ends at time $t$ 
	- Using the chain rule, we can rewrite 	
		$$
		P(\Theta|\phi,h) = P(\theta_{s_1, a_1}|\phi,h)P(\theta_{s_1, a_1}|\Theta_1,\phi,h)...P(\theta_{s_T, a_T}|\Theta_{T-1},\phi,h)P(\Theta \setminus \Theta_T|\Theta_T,\phi,h)
		$$
	-  $T$ is the length of the simulation
	- $\Theta \setminus \Theta_T$ is the (random) set of parameters that are not required for a simulation
- For each simulation $i$, we sample $P(\phi|h_t)$ at the root
	- Then lazily sample $\theta_{s_t, a_t}$ parameters are required, conditioned on $\phi$ and all $\Theta_{t-1}$ parameters sampled for the current simulation
- If the transition parameters for different states and actions are independent, we can completely forgo sampling a complete $\mathcal{P}$
	- Instead only draw necessary parameters individually for each state-action pair
## Rollout Policy Learning
- Choice of rollout policy $\pi_{ro}$ is important *if simulations are few*
	- **Otherwise simple *uniform random* policy could be used to provide estimates**
- Propose learning $Q_{ro}$
	- The optimal Q-value in the real MDP
	- Use a model-free approach (e.g., Q-learning) to learn from samples $(s_t,a_t,r_t,s_{t+1})$ obtained [[Reinforcement Learning#Off-policy methods|off-policy]] as a result of the interaction of Bayesian agent with the environment
	- **Use $\epsilon$-greedy policy with respect to $Q_{ro}$ as rollout policy $\pi_{ro}$**
# Evaluation
- Evaluation quantified based on reward obtained
- Compared against other algorithms on a set of standard problems
- Showcase scalability using an infinite 2D grid with complex correlation between reward locations

# Implementation
- https://github.com/acguez/bamcp.git
# Notables
todo...
# Future Work
todo...


