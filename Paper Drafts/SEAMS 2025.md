# Ideas:
1. **Deficiency in uncertainty-awareness of proactive self-protecting
	- Highlight missing uncertainty-awareness from existing literature
		- Papers to consider: 
			- **[[2022 - Robust Moving Target Defense against Unknown Attacks A Meta-Reinforcement Learning Approach - Li]]**
			- [[2020 - Multi-agent Reinforcement Learning in Bayesian Stackelberg Markov Games for Adaptive Moving Target Defense - Sengupta]]
	- Propose solution using Bayesian modelling and algorithms
		- **Incorporate BAMCP ([[2012 - Efficient Bayes-Adaptive Reinforcement Learning using Sample-Based Search - Guez|ref]]) to training stage of Metal-RL based solution**
	
	- Potential evaluation scenarios:
		- The attacker type changes (potentially multiple times) over the course of an experiment
		- The provided attacker type probability distribution of is incorrect
		- The provided attacker type probability distribution of changes over time
		- The reward function is designed incorrectly?
			- i.e., Impact Score used to calculate reward does not align with attacker's perceived value
		- The success rate is designed incorrectly?
		- e.g., should not be 0.1 x ES but rather 0.01 x ES


2. ~~Compare metrics for quantitative evaluation of self-protecting solutions~~
	- ~~Use "surprise" metric from Bencomo's work?~~
	- Use NLR from [[2017 - A Game Theoretic Approach to Strategy Generation for Moving Target Defense in Web Applications - Sengupta]]?


# Title
Combining Bayesian Planning and Meta-Reinforcement Learning to Enhance Uncertainty-Awareness in Self-protecting Software
# Abstract
In the face of increasingly adaptive and stealthy adversaries, proactive self-protecting software, such as Moving Target Defense (MTD), offer strategies to enhance system security by dynamically altering system configurations, such as utilizing different technology stacks. However, existing approaches often require unrealistic assumptions in their knowledge about adversaries when devising their solutions. This paper proposes a novel framework combining Bayes-Adaptive Monte-Carlo Planning (BAMCP) with Meta-Reinforcement Learning (Meta-RL) to address challenges related to uncertainty when planning and adapting against unknown attack strategies. BAMCP is employed during the training phase to explore the landscape of potential attack scenarios systematically and derive potential robust policies. These policies form the initialization of a meta-policy (i.e., policy that can be fined-tuned to new scenarios) framework, which is capable of rapid adaptation using limited real-world samples during deployment. The proposed approach leverages BAMCP’s capability in handling uncertainty alongside Meta-RL’s strength in online adaptation, to achieve a balance between exploration and exploitation. Experimental results in simulated MTD environments demonstrate improvements in defense performance against both known and unknown attack strategies, highlighting the potential of this integrated approach for advancing self-protecting systems.


