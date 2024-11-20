#paper #gametheory 
**author:** Paruchuri
**conference:** AAMAS
**year**: 2008
**file:** [[2008-AAMAS-playing-games-for-security-solve-Bayes-stackelberg-games-Praveen.pdf]] 
# Summary
Propose scalable solution for solving [[Bayesian Stackelberg Game]] that does not require [[Game Theory#Harsanyi Transformation|Harsanyi transformation]] by exploiting the property that ***followers are independent of each other***. Instead, DOBSS operates directly on the compact Bayesian representation of the problem. 

# Approach
## Mixed-Integer Quadratic Program [[Linear Programming#Mixed Integer Quadratic Program (MIQP)|(MIQP)]]
- Begin with **case of single follower**; leader is row player and follower the column player
- $x$ is the leader's policy, vector of leader's available actions (i.e., pure strategies)
	- $x_i$ is the proportion of times in which an action (pure strategy) is used in the policy
- $q$ is the follower's policy
- $X$ and $Q$ are index sets of the leader and follower's actions, respectively
- $R$ and $C$ are payoff matrices for the leader and follower
	- $R_{ij}$ is the reward of the leader when the leader takes pure strategy $i$ and follower takes pure strategy $j$
	- $C_{ij}$ defined similarly but for follower
- First fix the policy of the leader to some policy $x$; then, ***formulate the optimization problem the follower solves*** to find its optimal response to $x$
$$ 
\begin{gather}
\max_{q} \sum_{j \in Q} \sum_{i \in X} C_{ij}x_i q_j \\ 
\text{s.t.} \; \sum_{j \in Q} q_j = 1 \\
\; q_j \geq 0
\end{gather}
$$
^Eq1
- This objective function maximizes the follower's expected reward given $x$
- The [[Linear Programming#Dual Problem|dual problem]] for this is given by,
$$
\begin{gather}
\min_a a  \\
\text{s.t.} \; a \geq \sum_{i \in X} C_{ij} x_i \; j \in Q
\end{gather} 
$$
^Eq2
- This dual problem has the same optimal solution value; linear programming optimality conditions characterize the optimal solution to the follower's problem with: the primal feasibility constraint in [[#^Eq1]], dual feasibility constraints in [[#^Eq2]], and complementary slackness,
$$
q_j(a - \sum_{i \in X} C_{ij}x_i) = 0 \quad j \in Q
$$
- The above conditions show that the follower's maximum reward value $a$ is the value obtained for every [[Game Theory#Pure Strategy|pure strategy]] with $q_j >0$ (i.e., each of these pure strategies is optimal) 
- The *leader seeks the solution $x$ that maximizes its own payoff*, given the follower uses an optimal response $q(x)$
$$
\begin{aligned}
\max_x & \sum_{i \in X} \sum_{j \in Q} R_{ij} x_i q_j\\
\text{s.t.} & \sum_{i \in X} x_i = 1 \\
& \sum_{j \in Q} q_j = 1 \\
& 0 \leq (a-\sum_{i \in X} C_{ij}x_i) \leq (1-q_j)M \\
& x_i \in [0...1] \quad \forall i \in X\\ 
& q_j \in \{0,1\} \quad \forall j \in Q\\
& a \in \mathbb{R}
\end{aligned}
$$^Eq3
- $M$ is some large constant; $a$ is the follower's maximum reward value as defined in [[#^Eq2]]
- $0 \leq (a-\sum_{i \in X} C_{ij}x_i) \leq (1-q_j)M$ enforces [[Linear Programming#Dual Problem|dual]] feasibility of the follower's problem (left inequality) and the [[Linear Programming#Complementary Slackness|complementary slackness]] constraint for an optimal pure strategy $q$ for the follower (right inequality) 
- [[#^Eq3]] is a integer program with non-convex quadratic objective
	- problem arises when applying [[Game Theory#Harsanyi Transformation|Harsanyi transformation]] due to large number of joint actions of follower and nonlinear objective function

## Decomposed MIQP


# Evaluation

# Implementation

# Notable Results

# Future Work



