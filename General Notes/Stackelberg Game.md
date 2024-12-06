#general #gametheory 

- sequential game where players are divided into *leaders* and *followers*
- leaders make the first move to maximize reward; followers respond accordingly to maximize their own

# Stackelberg Equilibrium
- Consider a two-player [[Game Theory#Normal-form Game|normal-form]], [[Game Theory#General-sum Game|general-sum]] game, where there is a leader $l$ and a follower $f$ 
$$
A_l \; \text{is the finite set of actions the leader has}
$$
$$
A_f \; \text{is the finite set of actions the follower has}
$$
$$ 
\Delta^l \; \text{is the set of probability distributions over the leader's actions}
$$
$$ 
\Delta^f \; \text{is the set of probability distributions over the follower's actions}
$$
$$
u_i(a_l, a_f) \; \text{for} \; i \in {l, f} \; \text{is the utilities for the players}
$$
with slight abuse in notation,
$$
u_i(x,y) = \mathbb{E}_{a_l \sim x,a_f \sim y}[u_i(a_l, a_f)]
$$
where $x\in \Delta^l$, $y \in \Delta^f$ are probability distributions over $A_l$ and $A_f$ 
- In general, we assume leader commits to a strategy $x \in X$, and given such an $x$, the follower chooses their strategy from their ***best-response (BR)*** set,
$$
BR(x) = \arg \max_{y\in \Delta^f} u_f(x,y)
$$
- The goal of the leader is to chose a strategy $x$ maximizing their utility subject to the follower's best response,
$$
\max_{x\in \Delta^l} u_l(x,y) \; s.t. \; y \in BR(x)
$$
- ==**There is a problem with the above optimization problem; $BR(x)$ may be *set valued* and $u_l(x,y)$ would generally differ depending on which $y \in BR(x)$ is chosen==**
## Strong Stackelberg Equilibrium (SSE)
- Assume the follower breaks ties in favour of the leader
- In this case, the optimization problem becomes,
$$
\max_{x\in \Delta^l,y\in BR(x)} u_l(x,y)
$$
- This is the most *optimistic* variant
- **Always guaranteed to exists** 
## Weak Stackelberg Equilibrium (WSE)
- Assume follower breaks ties in adversarial manner
- This is the most *pessimistic* variant

$$
\max_{x\in \Delta^l} \min_{y \in BR(x)} u_l(x,y)
$$
- **Not always guaranteed to exists**


# Bayesian Stackelberg Game
- Combination of [[Bayesian Game]] and Stackelberg Game
- Consider a Bayesian game with a set of $N$ agents, where each agent $n$ belongs to a given set of types $\theta_n$
## Security Game
- For example, consider a game with two agents - leader type $\theta_1$ and and follower type $\theta_2$ 
- There is only 1 leader type; but there are *multiple* follower types
- **The leader does not know the follower's type**
- For each agent $n$, there is a set of strategies $\sigma_n$  and a utility function $u_n : \theta_1 \times \theta_2 \times \sigma_1 \times \sigma_2 \longrightarrow \mathbb{R}$ 
- ***Goal***: **find the optimal strategy for the leader to commit to; given the follower may know this mixed strategy when choosing their own strategy**
- Bayesian games can be transformed into [[Game Theory#Normal-form Game|normal-form game]] using [[Game Theory#Harsanyi Transformation|Harsanyi transformation]]
	- After transforming, existing linear programming-based methods can be used to find the optimal strategy
	- Harsanyi transformation involve introducing chance node that determines the follower's type; transforming the leader's [[Game Theory#Incomplete Information Game|incomplete information]] regarding follower into [[Game Theory#Imperfect Information Game| imperfect information]].
	- Set of actions for leader player stays the same
	- Set of actions for follower becomes cross product of each follower type's set of possible actions (grows exponentially)
	