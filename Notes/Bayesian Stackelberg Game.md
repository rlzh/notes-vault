#general #gametheory 
- Combination of [[Bayesian game]] and [[Stackelberg game]]
- Consider a Bayesian game with a set of $N$ agents, where each agent $n$ belongs to a given set of types $\theta_n$
# Security Game
- For example, consider a game with two agents - leader type $\theta_1$ and and follower type $\theta_2$ 
- There is only 1 leader type; but there are *multiple* follower types
- The leader does not know the follower's type
- for each agent $n$, there is a set of strategies $\sigma_n$  and a utility function $u_n : \theta_1 \times \theta_2 \times \sigma_1 \times \sigma_2 \longrightarrow \mathbb{R}$ 
- ***Goal***: **find the optimal strategy for the leader to commit to; given the follower may know this mixed strategy when choosing their own strategy**
- Bayesian games can be transformed into [[Game Theory#Normal-form Game|normal-form game]] using [[Game Theory#Harsanyi Transformation|Harsanyi transformation]]
	- After transforming, existing linear programming-based methods can be used to find the optimal strategy
	- Harsanyi transformation involve introducing chance node that determines the follower's type; transforming the leader's [[Game Theory#Incomplete Information Game|incomplete information]] regarding follower into [[Game Theory#Imperfect Information Game| imperfect information]].
	- Set of actions for leader player stays the same
	- Set of actions for follower becomes cross product of each follower type's set of possible actions (grows exponentially)
	