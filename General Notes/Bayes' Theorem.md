#general #bayesian 

- Gives a mathematical rule for inverting [[Statistics#Conditional Probability|conditional probabilities]] 
- Allows finding the probability of a cause given its effect
	- e.g., if the risk of developing health problems is known to increase with age, Bayes' theorem allows the risk to an individual of a known age to be assessed more accurately by conditioning it relative to their age.
$$
P(A|B) = \frac{P(B|A)P(A)}{P(B)}
$$
where, 
- $A$ and $B$ are events, and $P(B) \neq 0$ 
- $P(A|B)$ is a [[Statistics#Conditional Probability|conditional probability]] called the **posterior probability**
- $P(B|A)$ is also a conditional probability called the **likelihood**
	- likelihood of $A$ given fixed $B$: $P(B|A) = L(A|B)$
- $P(A)$ is called **prior probability**
- $P(B)$ is called **marginal probability (or evidence)**
# Proof
- Derived from the [[Statistics#Conditional Probability#Definition|definition]] of conditional probability
- Given, 
$$
P(A|B) = \frac{P(A\cap B)}{P(B)}, \text{if} \; P(B) \neq 0
$$
and
$$
P(B|A) = \frac{P(A\cap B)}{P(A)}, \text{if} \; P(A) \neq 0
$$
Solving for $P(A\cap B)$ gives,
$$P(A \cap B) = P(B|A) P(A)$$
and substituting into the expression for $P(A|B)$ gives Bayes' theorem.