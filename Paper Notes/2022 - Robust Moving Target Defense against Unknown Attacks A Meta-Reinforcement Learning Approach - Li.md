**author:** Li
**conference/journal:** Decision & Game Theory for Security
**year**: 2022
**file:** [[2022-DecisionGTforSec-robust-MTD-against-unknown-attacks-meta-RL-approach-Li.pdf]]
# Summary
todo...
# Approach
## System model
- Each state is a system config
- Defender chooses next system config at each time step (deterministically)
- There is a cost to migrate between configs
- There is loss if system is compromised 
- Assume compromise is immediately known at end of time step (and  system is recovered in next time step)
## Threat Model
- Attacker chooses system config to attack (after defender chooses)
- Attacker succeeds with certain prob. if correct system config selected
- Attacker always learns system config at end of each time step
- True attack type is initially unknown to defender (but has estimate of possible attacks)
## Two-stage defence
- Pre-train a meta-policy on a variety of attacks in simulated environment
- consider worst-case scenario and "weak" attacker scenario
- important observation: defender optimal policy can be formulated as single-agent MDP
- assume defender commits to a stationary policy first, then attacker chooses a policy  over a set of response policies
- In MDP:
	- state space is same as MG
	- action space is based on policies (i.e. probability vectors)
	- state transition becomes stochastic
## Meta RL
- view defender problem of solving its MDP against a particular attack as a task
- input to algorithm includes a distribution of attacks estimated from public datasets
- defender's policy represented by a parametrized function (e.g., neural network)
- in each iteration, a batch of K attacks are sampled from attack distribution
- each sampled attack generates a trajectory, which is used to compute a meta-policy 
![[reptile-meta-RL-for-MTD-algo.png]]
# Evaluation
- Evaluation setup in same scenario as [[2017 - A Game Theoretic Approach to Strategy Generation for Moving Target Defense in Web Applications - Sengupta]]
	- 3 different types of potential attackers
	- Same configurations space for defender
- Only proposed approach compared against uniform random strategy (URS) for defender 
- There are some variations for attacker: URS, Best Response strategy, and Worst Response strategy
- Evaluation metrics based on total loss (i.e., utility)
# Implementation
todo...
# Notables
todo...
# Future Work
todo...


