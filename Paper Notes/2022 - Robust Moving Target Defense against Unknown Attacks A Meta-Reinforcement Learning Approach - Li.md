#paper #mtd #RL #security 
**author:** Li
**conference/journal:** Decision & Game Theory for Security
**year**: 2022
**file:** [[2022-DecisionGTforSec-robust-MTD-against-unknown-attacks-meta-RL-approach-Li.pdf]]
# Summary
- Propose two stage meta-[[reinforcement learning]] (RL) based MTD framework
- Show that two player MTD game (solving for [[Stackelberg Game#Strong Stackelberg Equilibrium (SSE)|SSE]]) can be reduced to solving single-agent MDP
## Motivation
- Other solutions  assume attacker type (including physical, cognitive, computational abilities and constraints) **is known or sampled from a known distribution** (this is rarely true in practice)
	- Could resolve this by using fully online approach where defender assumes zero prior knowledge; but this approach requires collecting a large amount of samples covering various attacks, which is typically infeasible
## Proposed approach
- 2 stage meta-RL based MTD framework
	- Training stage: a meta-MTD policy is learned by solving multiple Stackelberg Markov games using experiences sampled from set of possible attacks
	- Test stage: adapt meta-policy based on samples collected on the fly
- Assumes defender has rough estimate of possible attacks
# Approach
- Attacker and defender share same state space; both know true state at beginning of time t; but defender acts first
- Action space is joint between attacker action space and defender action space
- Attacker action space and defender action space is the same as state space
- State transition is deterministically determined by the defender’s action
- Assumes stationary policies
- Consider [[Stackelberg Game#Strong Stackelberg Equilibrium (SSE)]] as solution concept
## System model
- System state at any time is based on its configuration; $N$ configurations means $N$ states
	- Each configuration consists of multiple adjustable parameters across different layers of system
	- $s^t$ denotes the configuration of the system at time $t$
- Defender is leader; attacker is follower; i.e. defender chooses action first
	- Defender chooses the next configuration according to policy $\pi^t_{D}$
	- Defender policy should be *randomized* (to increase attacker's uncertainty)
- Defender has cost for each migration 
	- ***Migration cost*** is denoted as $m_{ij} \geq 0$  to move from configuration $i$ to configuration $j$
- Defender also incurs cost for compromise
	- ***Compromise cost is denoted as $l_{s^t}$
	- This cost includes the cost to recover from potential damages
	- Assumes the defender can discover whether system is compromised or not at the end of each time step; and recovers the system if it is compromised; but this can be generalized to settings where feedback is delayed or imperfect
- **Defender goal is to minimize expected total cost**: *cost from migration and cost from compromise*
- Assumes: *Defender can discover whether system is compromised or not at end of time step; and recover the system if compromised*
	- Model can be generalized to reflect delay in feedback on compromises
## Threat Model
- Assumes: *Attacker always learns the system’s defense configuration at the end of each time step*
- Attacker chooses a system configuration to attack at beginning of each time step 
	- Denoted as $\tilde{s}^{t}$ 
	- Attack fails if attacker chooses wrong system configuration; $\tilde{s}^t\neq s^t$ 
	- Attack succeeds with some probability $\mu_{s^t}$ otherwise, and fails with probability $1-\mu_{s^t}$ 
	- Successful attack lead to a loss for the defender
- **Attack type** modelled based on a tuple $(\mu,\textbf{l})$ 
	- $\mu$ is the vector of attack success rates over a set of configurations
		- This is modelled based on the Exploitability Score from [[#CVSS]]
	- $\textbf{l}$  is the vector losses for all configurations
		- This is modelled based
	- These values can be derived from real measurements or publicly available database, such as National Vulnerability Database (NVD)
- An **attack** is defined as $\xi = (\mu, \textbf{l}, \pi_\mathcal{A})$ 
	- The attack includes its *type and policy*
- An attacker's policy is considered to be a randomized stationary policy 
	- $\pi_\mathcal{A}: S \rightarrow \Delta(S)$, which can be equivalently defined as the matrix $\textbf{q}$ where $q_{ij}$ denotes the probability of attacking configuration $j$ if the system is in configuration $i$ in the previous time step
## Markov Game Model for MTD
- Model the sequential interaction between defender and attacker of a given type as two-player [[Game Theory#General-sum Game|general-sum]] [[Reinforcement Learning#Markov Property|Markov]] Game (MG) denoted by
		$$G=(S,A,\mathcal{P},r,\gamma)$$
	- $S$ is the state space 
		- Both defender and attacker know the true system configuration $s^{t-1}$ at the beginning of time step $t$
	- $A=A_\mathcal{D} \times A_\mathcal{A}$ is the joint action space for the defender and attacker, respectively
		- Assume $A_\mathcal{D} = A_\mathcal{A} = S$ (i.e., defender picks the next configuration and attacker picks the next configuration to attack)
		- $a^t = (s^t, \tilde{s}^t)$ denotes the joint action at time $t$
	- $\mathcal{P} = S \times A \rightarrow \Delta(S)$ is the state transition function that represents the probability of reaching a state $s' \in S$ given current state $s \in S$ and the joint action $a \in A$ 
		- State transition is ***deterministic*** as next state is completely determined by defender's action
	- $r=\{ r_\mathcal{D}, r_\mathcal{A}\}$ where $r_{\mathcal{D}}: S \times A \rightarrow \mathbb{R}_{\leq 0}$ and $r_{\mathcal{A}}: S \times A \rightarrow \mathbb{R}_{\geq 0}$ are the reward functions for the defender and attacker, respectively
		- $r_\mathcal{D} = r_{\mathcal{D}}(s^{t-1}, a^t)=-\textbf{1}_{s_t=\tilde{s}^t} \mu_{s^t}l_{s^t} - m_{s^{t-1}{s^t}}$, where $\textbf{1}_{(.)}$ is the indicator function
		- $r_\mathcal{A} = r_{\mathcal{A}}(s^{t-1}, a^t)=-\textbf{1}_{s_t=\tilde{s}^t} \mu_{s^t}l_{s^t}$
	- $\gamma \in (0,1]$ is the discount factor 
- The total expected return is given by,
	$$V^{\pi_\mathcal{D},\pi_\mathcal{A}}_i(s^0) = \mathbb{E}_{\pi_\mathcal{D},\pi_\mathcal{A}}[\sum^\infty_{t=0} \gamma^t r_i (s^{t-1}, a^t|s^0)] $$
	where $i \in {\mathcal{D} ,\mathcal{A}}$
- Goal of each player is to maximize its expected return
- The approach considers [[Stackelberg Game#Stackelberg Equilibrium|Stackelberg Equilibrium]] (specifically [[Stackelberg Game#Strong Stackelberg Equilibrium (SSE)|Strong Stackelberg Equilibrium (SSE)]]) 
	- For each defense policy $\pi_\mathcal{D}$, let $B(\pi_\mathcal{D})$ denote the set of attack policies that maximizes $V^{\pi_D,\cdot}_\mathcal{D}(s^0)$ for any $s^0$ 
		$$
		B(\pi_\mathcal{D}) = \{\pi_\mathcal{A}: V^{\pi_\mathcal{D},\pi_\mathcal{A}}_\mathcal{A} = \max_{\pi'_\mathcal{A}} V^{\pi_\mathcal{D},\pi'_\mathcal{A}}_\mathcal{A} (s^0), \forall s^0 \in S\}
		$$
	- A pair of *stationary policies* $(\pi^*_\mathcal{D}, \pi^*_\mathcal{A})$ forms a SSE if for any $s^0 \in S$,
		$$
		V^{\pi^*_{\mathcal{D}},\pi^*_\mathcal{A}}_{\mathcal{D}} =\max_{\pi_\mathcal{D},\pi_\mathcal{A}\in B(\pi_\mathcal{D})} V^{\pi_{\mathcal{D}},\pi_\mathcal{A}}_{\mathcal{D}}(s^0)
		$$
	- SSE for above Markov game is guaranteed to exist due to finite state and action spaces and deterministic transition 
## Two-stage Defense
- [[#Markov Game Model for MTD|Markov Game formulation]] considers worst-case scenario; assumes attacker is smart
	- However solution obtained may be conservative in face of “weaker” attack who does not respond to defense strategically
	- **Exact type of attacker also is required**; or else solution may be poor
- Proposed solution:
	- Pre-train a meta-policy on a variety of attacks in simulated environment (using attack types for existing databases and penetration testing tools)
	- At test stage, the learned meta-policy is applied an updated using feedback (rewards) received in face of real attacks (not necessarily in training set)
## Reduce Markov Game into MDP
- Begin by generalizing and allowing the attacker to respond to defense policy in sub-optimal way
	- This allows incorporating diverse attack behaviour ("weaker" attacks) into meta RL
- Assume defender commits to a stationary policy $\pi_\mathcal{D}$
	- The attacker chooses a policy $\textbf{q}_s \in R(s,\pi_\mathcal{D})$  for any $s$, where $R(\cdot, \pi_\mathcal{D})$ denotes a **function that** **provides** **a set of response policies for the attacker**
- A pair of policies $(\pi_\mathcal{D},\pi_\mathcal{A})$ is a ***generalized SSE*** if $\pi_\mathcal{D}$ minimizes the defender's expected loss, assuming the attacker responds according to its response set $R$ in favour of the defender when there is a tie
## Assumptions
1. For any state $s \in S$ the attacker's response set $R$ is either a singleton or all the responses are equally good to the defender
	- *This allows the defender to infer the attack policy under each state*
2. For any state $s \in S$, the attacker's policy $\textbf{q}_s$ in state s only depends on $s$ and the defender's policy at state $s$ denoted $\textbf{p}_s$, and is independent of the defender's policy in other states
	- *This allows the defender's reward at any time to depend on the current state and defense action only, but not future states*
## Lemma 2
When [[#Assumptions]] hold, the optimal defense policy in the sense of a generalized SSE can be found by solving a single-agent MDP with continuous actions for the defender
- **Proof:** consider the following MDP for the defender $(S, A', \mathcal{P}', r', \gamma)$ which is derived from the earlier [[#Markov Game Model for MTD]]
	- $S$ is the same state space (i.e., each state is a configuration)
	- $A' = \Delta(S)$ is the action space of the defender
		- We redefine the defender's action in configuration $s$ as the probability vector (i.e., policy) $\textbf{p}_s$
	- $\mathcal{P}': S \times A' \rightarrow \Delta(S)$ is the state transition function
		- The probability of reaching configuration $s'$ from previous state $s$ after defender action $\textbf{p}_s$ is $\mathcal{P}'(s'|s,\textbf{p}_s)=p_{ss'}$
		- **state transition is now *stochastic*** due to the fact the defender's actions are now probabilistic
	- $r': S \times A' \rightarrow \mathbb{R}_{\leq 0}$ is the reward function for the defender
		- Note: the attacker's policy can be represented as $\textbf{q}_{s^{t-1}} = R(s^{t-1}, \textbf{p}_{s^{t-1}})$ (based on [[#Assumptions]])
		- $r'(s^{t-1}, \textbf{p}_{s^{t-1}}) = \sum_{s^t} p_{s^{t-1}s^t} q_{s^{t-1}\tilde{s}^t}r_\mathcal{D}(s^{t-1}, a^t)$ defines the defender's reward
		- The reward function only now only depends on the defender's action (instead of the joint action of the attacker and defender in the Markov Game)
	- $\gamma$ is the same discount factor
- Finding the best defense policy can be reduced to finding a deterministic policy $\pi: S \rightarrow A'$ that maximizes the total expected return of the MDP,  $\mathbb{E}[\sum^\infty_{t=0}\gamma^t r'(s^{t-1}, \textbf{p}_{s^{t-1}}|s^0)]$ 
...
## Robust Defense via Meta-RL
- Adopt **Reptile**, which is a first-order model agnostic metal-learning algorithm 
- View defender's problem of solving its MDP against a particular attack denoted, $\xi^k=(\mu_k,\textbf{l}_k, R_k)$ 
	- $R_k$ is the response function defined in [[#Reduce Markov Game into MDP]] 
	- The input to the algorithm includes a **distribution of attacks** denoted $\mathcal{P}(\xi)$ and two step sizes $\alpha$ and $\beta$ 
- Consider th defender's policy as a parameterized function (e.g., a neural network) $\pi_\mathcal{D}(\theta)$ with parameters $\theta$ 
- Algorithm starts with initial model $\theta^0$
- Over $T$ iterations, the algorithm samples a batch of $K$ attacks from $\mathcal{P}(\xi)$ in each iteration
	- For each attack, a trajectory of length $H$ is generated and used to compute the meta-policy $\theta^{t}_k$ using gradient descent with the following loss function
	$$
	\mathcal{L}_{\xi^k}(\pi_\mathcal{D}(\theta)) = -\mathbb{E}_{\pi_{\mathcal{D}}(\theta)}[\sum^{H}_{t=1}r'(s^{t-1}, \textbf{p}_{s^{t-1}})]
	$$
	- Set $\gamma = 1$ here such such that defender is allowed to query a limited number of samples within the fixed horizon $H$
	
![[reptile-meta-RL-for-MTD-algo.png]]
# Evaluation
- Evaluation setup is similar to scenario in [[2017 - A Game Theoretic Approach to Strategy Generation for Moving Target Defense in Web Applications - Sengupta]]
	- 3 different types of potential attackers
	- Same configurations space for defender
	- But difference in determining reward 
- **Attack types:**
	- Derived from [[CVSS]] 
	- An attack with Impact Score (IS) $\in [0,10]$ and Exploitability Score (ES) $\in [0, 10]$ will generate **$10 \times$ IS unit time loss and have $0.1 \times$ ES success rate** ([[2022-DecisionGTforSec-robust-MTD-against-unknown-attacks-meta-RL-approach-Li.pdf#page=12&selection=41,4,60,22|quote]])
- **Comparison Baselines**
	- Defender:
		- Uniform Random Strategy: defender uniformly samples a configuration to switch to at each time step
		- 
- There are some variations for attacker: URS, Best Response strategy, and Worst Response strategy
- Evaluation metrics based on total loss (i.e., utility)

![[robust-mtd-meta-rl-experiment-params.png|500]]
# Implementation
- [https://github.com/HengerLi/meta-RL](https://github.com/HengerLi/meta-RL) 
- Experiments done using OpenAI gym + PyTorch + stable baseline 3
# Notables
- Does not address noisy/delayed feedback (i.e. partially observable)
- Modelling is done differently (based on MTD configurations instead of kill-chain)

- Question
	- Why does state transition in MDP formulation become stochastic? ([[#Lemma 2|ref]])
	- How does one determine $\textbf{q}_{s^{t-1}}$ in [[#Lemma 2]]?
	- How does one later use the meta-policy obtained from [[#Robust Defense via Meta-RL]]?

- Source of uncertainty is:
	- Regarding probability distribution of attacker type (this is assumed to be inaccurate)
	- Some parts of solution is still assumed 
		- An attack with Impact Score (IS) $\in [0,10]$ and Exploitability Score (ES) $\in [0, 10]$ will generate **$10 \times$ IS unit time loss and have $0.1 \times$ ES success rate** ([[2022-DecisionGTforSec-robust-MTD-against-unknown-attacks-meta-RL-approach-Li.pdf#page=12&selection=41,4,60,22|quote]])
# Future Work
todo...


