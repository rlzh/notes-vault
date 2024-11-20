#paper #security #mtd #gametheory #bayesian #uncertainty

- **author**: Sengupta
- **conference**: AAMAS
- **year**: 2017
- **file**: [[2017-AAMAS-GT-approach-to-strat-gen-for-MTD-in-web-apps-Sengupta.pdf]]
# Summary
Propose game theory-based modelling of MTD for web applications. The MTD technique adjust the attack surface by switching between different configurations of the web applications, namely by switching the language the application is running with or the database the application is using. 

The approach also considers and captures uncertainty regard the *type* of attacker targeting the system with the use of Bayesian Stackelberg Game modelling. 

Evaluation is done on simple scenario with only 4 different configurations, 3 types of different attacker types, and ~300 attacks.

# Approach
- Models MTD scenario as [[Bayesian Stackelberg Game|Bayesian Stackelberg Game (BSG)]] (i.e., combination of [[Stackelberg Game]] and [[Bayesian Game]]) ^c18c3c
 - a web app has a set of configs (tech stacks) (e.g., (Python, MySQL), (Java, Postgres), etc.)
- attacks on system defined based on defined based on public dataset (NVD 2013-2016)
- switching between configs has a cost (similar configs have lower cost; different configs have higher cost)
- defender needs to decide next valid config given *current config*
- states modelled based on configurations available to the defender
$$
C=C_1 \times C_2 \times C_3 \times ... \times C_n \quad \text{where} \; C_i \; \text{is set of technologies available in} \; i^{th} \; \text{layer}
$$
## Game Theoretic Modelling
- 2 types of agents; defender type only has 1 type; attacker type has finite amount
- each agent has a finite set of actions
$$
\theta_i \; \text{is set of types for player} \; i \in \{1,2\}, \; \text{where 1 is defender and 2 is attacker}
$$
$$
|\theta_1| = 1
$$
for each attacker type:

$$
\theta_{2_{i}} = \textlangle\text{name}, \{(\text{expertise}, \; \text{technology}),...\}, P_{\theta_{2_{i}}} \textrangle
$$
$$
\text{expertise} \in [0,10],
$$
$$
P_{\theta_{2_{i}}} \;\text{is the defender's belief about attacker type} \; \theta_{2_{i}} \; \text{attacking their app}
$$
- Attacker agent's actions defined based on CVSS scores and estimated expertise
$$
A_{\theta_{i}} = \; \text{finite set of actions availlable to player} \; i
$$
$$
A_{\theta_1} = \; \text{defender action set is defined based on configurations available}
$$
$$
A_{\theta_{2}} = \; \text{attacker action set is defined based on \textit{all} attacks used by at least one attacker type}
$$
that is,
$$
\text{attack} \; a \in A_{\theta_{2}} \; \text{if it affects} \; \geq 1 \; \text{technology used in configuration}
$$
for attacker type $t$ and attack $a$,
$$
f(\theta_{2_{t}}, a) = 
\begin{cases}
	1, & \text{iff} \; T_t \cap T^a \neq \emptyset \; \text{and} \; ES_a \leq \text{expertise},\\
	0, & \text{otherwise}
 \end{cases}
$$
$$
T_t \; \text{is the set of technologies attacker has expertise in}
$$
$$
T^a \; \text{is the set of technologies affected by attack} \; a
$$
- Reward values based on [[CVSS#CVSS (Common Vulnerability Scoring System) v2.0 Equations|CVSSv2]] metrics
	- attacker has 0 reward for no-op
	- defender has 0 reward for defending
	- defender has honey-net config which can detect certain types of attacks (not actually implemented!)
	
- Switching cost between configuration is based on "rule of thumb" and estimated values
	- e.g., if no common technology is shared between configurations, then cost will be large
	- e.g., cost for switching between DBs is large due to data transfer

## Switching Strategy Generation
- Based on [[Stackelberg Game#Stackelberg Equilibrium#Strong Stackelberg Equilibrium (SSE)|Strong Stackelberg Equilibrium (SSE)]] 
- Solving problem is [[Problem Complexity#NP-hard|NP-hard]] due to multiple attacker types
#### Assumptions
- Attacker chooses [[Game Theory#Pure Strategy|pure strategy]](i.e., single attack action that maximizes reward)
- Pure strategy of an attacker is not influenced by the strategy of the other attacker types (attacker types attack selection is independent of the attack action chosen by another type)

- Solve for optimal mixed strategy using [[2008 - Decomposed Optimal Bayesian Stackelberg Solver (DOBSS)|DOBSS]]
- Formulate problem as [[Linear Programming#Mixed Integer Quadratic Program (MIQP)|Mixed Integer Quadratic Program (MIQP)]]
$$
\max_{x,n,v} \sum_{c\in C} \sum_{\theta_{2_i} \in \theta_2} \sum_{a \in A_{\theta_{2_i}}} P_{\theta_{2_i}} R^{D}_{a,\theta_{2_i},c} \; x_c n^{\theta_{2_i}}_a
$$
where,
$$ 
\begin{gather}
x \; \text{is mixed strategy for the defender} \\
n^{\theta_{2_i}} \; \text{is the pure strategies for each attacker type} \\
P_{\theta_{2_i}} \; \text{is the attacker type uncertainty (probability)} \\
R^{D}_{a,\theta_{2_i},c} \; \text{is the reward function for the defender}
\end{gather}
$$
- Incorporate [[McCormick envelopes]] to design convex function that estimates switching costs between configurations
$$
\begin{align}
\max_{x,n,v} \sum_{c\in C} \sum_{\theta_{2_i} \in \theta_2} \sum_{a \in A_{\theta_{2_i}}} P_{\theta_{2_i}} R^{D}_{a,\theta_{2_i},c} \; x_c n^{\theta_{2_i}}_a - \alpha \sum_{i \in C} \sum_{j \in C} K_{ij} w_{ij}
\end{align}
$$
where,
$$
\begin{gather}
K_{ij} \; \text{is the switching cost between configuration $i$ and $j$} \\
w_{ij} \; \text{is an approximate value of $x_i \cdot x_j$ ($x_i$ is the probability of selecting config $i$...)} \\
\text{with \textbf{additional constraints...}}
\end{gather}
$$
- Also need solve [[Linear Programming#Dual Problem|dual problem]] of maximizing rewards for each attacker type given the defender's strategy (this ensures the attackers select the best action) as part of constraints
$$
0 \leq v^{\theta_{2_i}} - \sum_{c \in C} R^{A}_{a,\theta_{2_i},c}x_c \leq (1-n^{\theta_{2_i}}_a)M
$$
where,
$$
\begin{gather}
v^{\theta_{2_i}} \in \mathbb{R} \; \text{is the rewards for each attacker type}\\
R^{A}_{a,\theta_{2_i},c} \; \text{is the reward function for attackers} \\
n^{\theta_{2_i}}_a \in \{0,1\} \\
M \; \text{is a large positive constant}
\end{gather}
$$
- To consider repeated games, $w_{ij}$ values need to be updated based on the previous round of the game
- The expression $w_{ij}=x_i \cdot x_j$ would need to become $x^{t}_{i} \cdot x^{t+1}_{j}$ where $x^{t}_{i}=1$  for the i-th configuration that was deployed at time $t$ and 0 for the others.
- The **final optimization problem** becomes,
$$
\max_{x,n,v} \sum_{c\in C} \sum_{\theta_{2_i} \in \theta_2} \sum_{a \in A_{\theta_{2_i}}} P_{\theta_{2_i}} R^{D}_{a,\theta_{2_i},c} \; x_c n^{\theta_{2_i}}_a - \alpha \sum_{i \in C} \sum_{j \in C} K_{ij}\cdot x^t_{i}\cdot x_j
$$
where,
$$
x^t_i \; \text{are constants from obtained from the previous round}
$$
# Evaluation
- Used simple example with 4 configurations and 3 attacker types to answer the following 3 questions:
1. Does [[#^c18c3c|BSG]]-based model generate better strategies than state-of-the-art?
2. Can approach compute set of critical vulnerabilities?
3. Can approach identify sensitive attacker types?
# Implementation
- GitHub repo: https://github.com/sailik1991/StackelbergEquilibribumSolvers 
# Notable Results

# Future Work