#security #mtd #gametheory #bayesian 
- **author**: Sengupta
- **conference**: AAMAS
- **year**: 2017
- **link**: [paper](obsidian://open?vault=thesis&file=Focus%2F2017-AMAS-GT-approach-to-strat-gen-for-MTD-in-web-apps-Sengupta.pdf)
# Summary
Propose game theory-based modelling of MTD for web applications. The MTD technique adjust the attack surface by switching between different configurations of the web applications, namely by switching the language the application is running with or the database the application is using. 

The approach also considers and captures uncertainty regard the *type* of attacker targeting the system with the use of Bayesian Stackelberg Game modelling. 

Evaluation is done on simple scenario with only 4 different configurations, 3 types of different attacker types, and ~300 attacks.

# Approach
- Models MTD scenario as Bayesian Stackelberg Game (i.e., combination of [[Stackelberg Game]] and [[Bayesian Game]])
 - a web app has a set of configs (tech stacks) (e.g., (Python, MySQL), (Java, Postgres), etc.)
- attacks on system defined based on defined based on public dataset (NVD 2013-2016)
- switching between configs has a cost (similar configs have lower cost; different configs have higher cost)
- defender needs to decide next valid config given *current config*
- states modelled based on configurations available to the defender
$$
C=C_1 \times C_2 \times C_3 \times ... \times C_n \quad \text{where} \; C_i \; \text{is set of technologies available in} \; i^{th} \; \text{layer}
$$
### Game Theoretic Modelling
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
\theta_{2_{i}} = <\text{name}, \{(\text{expertise}, \; \text{technology}),...\}, P_{\theta_{2_{i}}}>
$$
$$
\text{expertise} \in [0,10],
$$
$$
P_{\theta_{2_{i}}} \;\text{is the defender's belief about attacker type} \; \theta_{2_{i}} \; \text{attacking their app}
$$
- attacker agent's actions defined based on CVSS scores and estimated expertise
$$
A_{\theta_{i}} = \; \text{finite set of actions availlable to player} \; i
$$
$$
A_{\theta_1} = \; \text{defender action set is defined based on configurations availalbe (i.e., switching configuration)}
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
- reward values based on [CVSSv2](obsidian://open?vault=thesis&file=Notes%2FCVSSv2) metrics
	- attacker has 0 reward for no-op
	- defender has 0 reward for defending
	- defender has honey-net config which can detect certain types of attacks (not actually implemented!)
	
- switching cost between configuration is based on "rule of thumb" and estimated values
	- e.g., if no common technology is shared between configurations, then cost will be large
	- e.g., cost for switching between DBs is large due to data transfer

### Switching Strategy Generation
- based on [[Stackelberg Game#Stackelberg Equilibrium#Strong Stackelberg Equilibrium (SSE)|SSE]] 
- problem is [[Problem Complexity#NP-hard|NP-hard]] due to multiple attacker types
#### Assumptions
- attacker chooses pure strategy (i.e., single attack action that maximises reward)
- pure strategy of an attacker is not influenced by the strategy of the other attacker types (attacker types attack selection is independent of the attack action chosen by another type)

- solve for optimal mixed strategy using [[2008 - Decomposed Optimal Bayesian Stackelberg Solver (DOBSS)|DOBSS]]
- formulate problem as [[Mixed Integer Quadratic Program]] 
$$
\max_{x,n,v} \sum_{c\in C} \sum_{\theta_{2_i} \in \theta_2} \sum_{a \in A_{\theta_{2_i}}} P_{\theta_{2_i}} R^{D}_{a,\theta_{2_i},c} \; x_c n^{\theta_{2_i}}_a
$$
where,
$$ 
\begin{gather}
x \; \text{is mixed strategy for the defender} \\
n^{\theta_{2_i}} \; \text{is the pure strategies for each attacker type} \\
P_{\theta_{2_i}} \; \text{is the attacker type uncertainty (probability)}
R^{D}_{a,\theta_{2_i},c} \; \text{is the reward function for the defender}
\end{gather}
$$
- incorporate [[McCormick envelopes]] to design convex function that estimates switching costs between configurations
$$
\begin{align}
\max_{x,n,v} \sum_{c\in C} \sum_{\theta_{2_i} \in \theta_2} \sum_{a \in A_{\theta_{2_i}}} P_{\theta_{2_i}} R^{D}_{a,\theta_{2_i},c} \; x_c n^{\theta_{2_i}}_a - \alpha \sum_{i \in C} \sum_{j \in C} K_{ij} w_{ij}
\end{align}
$$
where,
$$
\begin{gather}
K_{ij} \; \text{is the switching cost between configuration $i$ and $j$} \\
w_{ij} \; \text{is an approximate value of $x_i \cdot x_j$ ($x_i$ is the probability of selecting config $i$)} \\
\text{with \textbf{additional constraints...}}
\end{gather}
$$
# Evaluation

# Implementation
- GitHub repo: https://github.com/sailik1991/StackelbergEquilibribumSolvers 
# Notable Results

# Future Work