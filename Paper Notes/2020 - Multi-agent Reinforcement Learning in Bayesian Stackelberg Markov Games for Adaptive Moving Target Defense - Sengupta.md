#paper #mtd #bayesian #gametheory #uncertainty #RL #sengupta
**author:** Sengupta
**conference/journal:** arXiv
**year**: 2020
**file:** [[2020-arxiv-mult-agent-RL-in-bayesian-stackelberg-markov-games-for-MTD-Sengupta.pdf]]
# Summary
Propose modelling MTD as [[#Bayesian Stackelberg Markov Game (BSMG)]] to capture uncertainty regarding type of attackers targeting the system. Uses two example scenarios (namely multi-configuration web app and defending against multi-stage attacks) to illustrate effectiveness of approach.
# Approach
- Motivation:
	- Markov Games (1) do not consider incomplete information about adversary and (2) consider weak thread model where the attacker has no information about defender's policy
	- [[Stackelberg Game#Bayesian Stackelberg Game|Bayesian Stackelberg Game (BSG)]] can only model single-stage games, which cannot be generalized to sequential decision making settings
## Bayesian Stackelberg Markov Game (BSMG)
- Extends BSG to *multi-stage sequential games*
- Goal is to use BSMG as unifying framework to characterize optimal movement policies, capture transition dynamics and costs, and improve reasoning against stronger threat models
- A BSMG can be represented by a tuple $(P, S, \Theta, A, \tau, U, \gamma^\mathcal{D}, \gamma^\mathcal{A})$ where,
	- $P=\{\mathcal{D}, \mathcal{A}=\{\mathcal{A}_1, \mathcal{A}_2, ... \mathcal{A}_t\}\}$  where $\mathcal{D}$ denotes the leader (defender) and $\mathcal{A}$ denotes the follower (attacker); only the follower has $t$ player types
	- $S = \{s_1, s_2, ... s_k\}$ are $k$ (finite) states of the game
	- $\Theta = \{\theta_1, \theta_2, ... \theta_k\}$ denote $k$ probability distributions for $k$ states over the $t$ attackers; where $\theta_i(s)$ denotes the probability of the attacker type being $\mathcal{A}_i$ in state $s$ 
	- $A = \{A^\mathcal{D}, A^\mathcal{A_1}, ... A^\mathcal{A_t}\}$ denotes the action sets of the player, where $A^i(s)$ represents the set of actions/pure strategies available to player $i$ in state s
	- $\tau^i(s, a^\mathcal{D}, a^\mathcal{A_i}, s')$ represents the state transition probability of reaching state $s' \in S$ from state $s \in S$ when the defender $\mathcal{D}$ chooses action $a^\mathcal{D}$ and attacker $\mathcal{A}_i$ chooses action $a^\mathcal{A_i}$ 
	- $U = \{U^\mathcal{D}, U^\mathcal{A_1}, ... U^\mathcal{A_t}\}$   where $U^\mathcal{D}(s, a^\mathcal{D}, a^\mathcal{A_i})$  and $U^\mathcal{A}(s, a^\mathcal{D}, a^\mathcal{A_i})$ represents the reward/utility of the defender $\mathcal{D}$ and attacker $\mathcal{A_i}$
	- $\gamma^i \in [0,1)$ is the discount factor for player $i$; assume $\gamma^\mathcal{D} = \gamma^\mathcal{A_i}$ 
- Defender expected to deploy a strategy first; and all attacker types know the defender's policy
- **Each individual stage games constitute [[Game Theory#Normal-form Game|normal-form]] [[Bayesian Game]] with distribution over attacker types**
	- Defender's [[Game Theory#Mixed Strategy|mixed policy]] denoted as $x$
	- Attacker type $\mathcal{A}_i$'s response set (set of best response to $x$) denoted as $R^i(x)$ 
	- If the response set for all attacker types is a singleton (only a single response), then the action profile $(x, R^1(x), ... R^t(x))$ constitutes a [[Stackelberg Game#Stackelberg Equilibrium|Stackelberg Equilibrium]] 
	- The approach suggests to go with [[Stackelberg Game#Strong Stackelberg Equilibrium (SSE)|Strong Stackelberg Equilibrium (SSE)]] as solution concept for BSMG
- For a given policy of the defender in BSMG, every attacker type will have a *deterministic* policy in all states $s \in S$ that is an optimal response
- For an SSE policy of the defender $x$,  each attacker type $\mathcal{A}_i$ has a deterministic best policy (action) $q_i$. 
	- The action profile $(x, q_1, ... q_t)$ denotes the SSE for the BSMG
- If an action profile $(x, q_1, ... q_t)$ yields the equilibrium values $V^D_{x,q}$ and $V^{A_i}_{x,q}$ for the players AND is an SSE of BSMG, then
	- $\forall s \in S$, $(x(s), q_1(s),...q_t(s))$ is an SSE of the bi-matrix [[Bayesian Game]] represented by the [[Reinforcement Learning#Q-values|Q-values]] $Q^{D,i}_{x,q_i}(s)$ and $Q^{\mathcal{A_i}}_{x,q_i}(s) \; \forall i \in \{1, ... ,t\}$ 

## Modelling 
### MTD for web-application security:
- Builds on approach presented in [[2017 - A Game Theoretic Approach to Strategy Generation for Moving Target Defense in Web Applications - Sengupta]]  
- BSMG has $|C|$ **states**, each representing a configuration of the MTD system
	- Each configuration has equal probability as starting state in an episode and no terminal state
- For each state, the **defender action set** is $A^\mathcal{D}(s) = C$ (the defender can choose to move to any configuration; including remaining in the current one)
- Three attacker types denoted as $\mathcal{A}=\{\mathcal{A_1}, \mathcal{A_2}, \mathcal{A_3}\}$ 
- **Attacker's action sets** represented by mined CVEs 
- Probability distribution of these attacker types in state $s$ denoted as $\theta_s$ 
	- ***This distribution remains the same in all states of the BSMG*** ([[2020-arxiv-mult-agent-RL-in-bayesian-stackelberg-markov-games-for-MTD-Sengupta.pdf#page=4&annotation=612R|quote]])
- Game starts in any of the states (configurations) at *random*
- ***Attacker reward*** uses [[CVSS]] Impact score as if the chosen attack is *expected* to work (i.e., if the configuration is vulnerable to attack) on the defense configuration 
- ***Defender cost*** appears to be is defined based on same [[#CVSS]] impact score (in the same manner as in [[2017 - A Game Theoretic Approach to Strategy Generation for Moving Target Defense in Web Applications - Sengupta]]) 
	- But *switching cost* is no longer part of defender reward function 
		- > In this setting, we do not capture the switching cost as part of the reward metric, but use it to guide the transition dynamics of the underlying environment [[2020-arxiv-mult-agent-RL-in-bayesian-stackelberg-markov-games-for-MTD-Sengupta.pdf#page=17&selection=90,4,91,100|2020-arxiv-mult-agent-RL-in-bayesian-stackelberg-markov-games-for-MTD-Sengupta, page 17]]
- Transition between states is **stochastic** ([[2020-arxiv-mult-agent-RL-in-bayesian-stackelberg-markov-games-for-MTD-Sengupta.pdf#page=18&selection=14,11,19,14|quote]])
	- Because transitions are now based on switching costs
	- Expensive switches between configurations have higher probability of failing
	- When a switch fails, the system remains in the same configuration
## Strong Stackelberg Q-learning
- Motivated by presence of (1) uncertainty in game parameters and (2) incomplete information regarding adversary
- Use multi-agent RL approach for [[#Bayesian Stackelberg Markov Game (BSMG)]], which considers a Bellman-style Q-learning approach for calculating agent policies over time
- **Adversary is simulated** and requires a simulation of the environment
![[bayesian-strong-stackelberg-q-learning.png|800]]
# Evaluation
- Evaluation is based on comparing defender reward values over episodes
- Not a lot of evaluation carried out overall (only 1 plot for each case study)
# Implementation
- Not available from paper
# Notables
- Novelty is in considering incomplete information about adversary (improve on Markov Games) and generalize to *sequential* decision making settings (improve on Bayesian Games)
- The **source of uncertainty** is 
	- Incomplete information about adversaries (by information, it means current *attacker type*)
	- i.e., uncertainty means we do not know the exact type of attacker we are dealing with in each state, *but* the **probability distribution of attacker types is assumed to be accurate**
- Expects to have a probability distribution $\theta_s$ over attacker types for each state $s$ of the game
	- **This $\theta_s$ stays the same for all $s$  in experiments** (may not be realistic?)
- State transition is ***stochastic***
- Defender rewards purely based on 
- Ultimately, this improves on [[2017 - A Game Theoretic Approach to Strategy Generation for Moving Target Defense in Web Applications - Sengupta]] by supporting *sequential decision-making* that is based on the current configuration of the system
		
# Future Work



