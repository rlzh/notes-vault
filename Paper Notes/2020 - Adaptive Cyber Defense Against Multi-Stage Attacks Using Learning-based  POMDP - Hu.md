#paper #mtd #pomdp #RL #bayesian #uncertainty 

**author:** Hu
**conference/journal:** ACM Transactions on Privacy and Security
**year**: 2020
**file:** [[2020-TransPrivSec-adaptive-defense-multi-stage-attack-using-POMDP-Hu.pdf]]
# Summary
- Paper novelty
1.  Deal with unknown state transition probabilities
2.  Deal with unknown utility functions
# Approach
- propose Thompson sampling-based RL + Q-learning
- maintain posterior distribution over transition probabilities; instantiate set of estimated transition probabilities from posterior distribution periodically
- consider scenario where transition probabilities are generated at start of attack and fixed, but unknown to defender

## Bayesian Attack Graphs (BAGs)
- weighted graphs where weights represent how likely exploits could succeed
- model multi-stage attacks that propagate through a network
- each node represents if the corresponding machine is compromised or not by attackers
- exploit probabilities quantified by CVSS exploitability metrics
- system states are stacked into vector
- assume attacker always tries to compromise all available machines; until all machines compromised
- a compromised machine in previous step stays compromised
- attacker will restart compromising machines from leaf nodes when there is no exploitable out-edges of compromised nodes
## POMDP Modelling
- defender actions include detection and reimage (assumes detection is implemented by manual/labour analysis! can only detect a small subset of machines in network!)
- example of confidentiality impact: # of suspicious read ops on machine i
- example of integrity impact: # of modified files on machine i
- example of availability impact: # of lost connections or remaining disk space on machine $i$
- defender aims to find policy to maximize aggregate utility
- focus on deterministic decisions
# Evaluation
- Evaluate on network with 10 machines
	- Each machine has its own set of vulnerabilities
- Evaluation outcomes primarily based on aggregate utility values plotted over increasing time steps
- Compared against: [[Reinforcement Learning#Value Iteration|value iteration]], uniform selection, [[Reinforcement Learning#Q-learning|Q-learning]] 
# Implementation
- Not available from paper
# Notables

# Future Work



