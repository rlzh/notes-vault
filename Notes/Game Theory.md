#general #gametheory 

# Incomplete Information Game
- Games where players lack knowledge about key aspects of the game (e.g., other player's payoffs or private information)
- Incomplete information is a broader concept that encompasses imperfect information (players may not even know the full rules of the game)

# Imperfect Information Game
- Games where players cannot fully observer all the actions taken by other players during the game; leading to uncertainty about their current state (round) of play

# Normal-form Game
- represent the game by way of a (payoff) matrix

|            | Left | Right  |
| ---------- | ---- | ------ |
| **Top**    | 4, 3 | -1, -1 |
| **Bottom** | 0, 0 | 3, 4   |
# Zero-sum Game
- a mathematical representation in game theory of a situation involving two entities where the result is an advantage of one side and an *equivalent* loss for the other
- i.e., one player's gain is equivalent to the other player's loss
- most often solved with the [[#Minimax Theorem]] Theorem
# General-sum Game
- non-zero sum game
# Pure Strategy
- provides a complete definition of how a player will play a game
- strategy is akin to an action
- i.e., deterministically selecting actions based on observations
# Mixed Strategy
- is an assignment of a probability to each pure strategy 
- There are infinitely many mixed strategies available to a player (since probabilities are continuous)
- Pure strategy can be considered as degenerate case of a mixed strategy, (i.e., a single pure strategy is selected with probability of 1 and every other strategy )
# Nash Equilibrium
- solution concept most commonly used for non-cooperative games
- If each player has chosen a strategy (i.e., action plan based on what has happened so far in the game) and no one can increase one's own expected payoff by changing one's strategy while other players keep theirs unchanged, then the current set of strategy choices constitutes a Nash equilibrium
- If two players Alice and Bob chooses strategies $A$ and $B$, $(A,B)$ is a Nash equilibrium if Alice has no other strategy available that does better than $A$ at maximizing her payoff in response to Bob choosing $B$, and vice versa for Bob.
- In a game in which Carol and Dan are also players, $(A,B,C,D)$ is a Nash equilibrium if $A$ is Alice's best response to $(B,C,D)$, etc.
- **There is a Nash equilibrium (possibly mixed strategy) for every finite game**


# Minimax Theorem
- Minimizes the worst-case potential loss
- States that for every finite, two-person [[#Zero-sum Game|zero-sum game]], there is a rational outcome
$$
\max_{x \in X} \min_{y \in Y}f(x,y) = \min_{y \in Y} \max_{x \in X}f(x,y)
$$
under certain conditions on sets $X$ and $Y$ and on the function $f$. Specifically,...



# Harsanyi Transformation
...