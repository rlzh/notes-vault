#general #bayesian #gametheory 
- Decision-making model which assumes players have **incomplete information**
- Players may hold private information 
- Payoffs may not be common knowledge
	- Players hold beliefs about payoff functions (beliefs are represented by probability distributions over possible payoff functions)
- A Bayesian game is defined by $(N, A, T, p, u)$
	- $N$ is the set of players within the game
	- $A$ is the action space for the game with $a_i \in A$ being the set of actions available to player $i \in N$
		- An ***action profile***, denoted $a=(a_1, a_2, ... a_N)$ is a list of actions (one for each player)
	- $T$ is the set of types of players, with $t_i$ being the types of player $i$
		- A ***type profile*** is denoted as $t=(t_1, t_2, ... t_N)$
	- $u=(u_1, u_2, ...,u_N)$ denotes the utility function of player $i$
	- $p(t) = p(t_1, ... , t_N)$ is the probability that player 1 has type $t_1$, etc.
# Beliefs
- In Bayesian games, player's about the game are denoted by a probability distribution over various types
- If players do not have private information, the probability distribution over types is known as a *common prior*
