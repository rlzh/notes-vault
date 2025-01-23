#general #RL #deeplearning #neuralnetwork #ML 

- Introduces deep [[Deep Learning#Neural Networks|neural networks]] to estimate policies and value-functions 
# Policy-based Method
- **Learn a [[Reinforcement Learning#Policy|policy]] function directly**
- Policy can be *deterministic* or *stochastic*
- Directly train policy (i.e., neural network) to select action to take given a state (or probability distribution over actions given state)
	- **No value function**
	- Policy is defined based on training the neural network
# Value-based Method
- **Learn a [[Reinforcement Learning#Value Function|value function]] that estimates the *expected value* of being in a state**
- Value refers to the [[Reinforcement Learning#Expected Return|expected return]] the agent can obtain if it starts in that state and acts according to a policy
- Indirectly obtain optimal policy by training a value function (i.e., neural network) that outputs the value of a state or state-action pair
	- Given the output, our policy will take action *e.g., greedily*
	- Policy is *not trained/learned*
	- Can learn two types of value-based functions: [[Reinforcement Learning#State-value function]] or [[Reinforcement Learning#Action-value function]] 

# Deep Q-Learning
- **Use a [[Deep Learning#Neural Networks|neural network]] that takes a state as input and outputs approximate Q-values for each action**
	- [[Reinforcement Learning#Q-learning|Traditional Q-learning]] is ***tabular***; meaning it uses arrays and tables to represent state-action values (i.e., **not scalable** when state-space and/or action-space is very big)
	![[deep-q-learning-concept.png|400]]
	
## Deep Q-Network
- **A neural network that is used to approximate the Q-values for each possible action at a state (i.e., value-function estimate)**
- Pre-processing input is important for certain problems
	- Helps reduce state-space (i.e., input space) complexity
	- e.g., for training in a video game environment, we can change input image from RGB to grey-scale
	- e.g., input could also be stacked to include *temporal* information (e.g., direction of object in game)
- Q-value estimation in [[Reinforcement Learning#Q-learning|Q-learning]] algorithm
	![[Q-learning-update.png|500]]
- Instead of updating Q-value directly like in [[Reinforcement Learning#Q-learning|Q-learning]], we define a **loss function** that compares the ***predicted Q-value*** and the ***Q-target*** using [[Machine Learning#Gradient Descent|gradient descent]] to update weights in the Deep Q-Network 
	- The **loss function** is defined as $y_j - Q(\phi_j, a_j; \theta)$, where $y_j = r_j + \gamma \max_{a'} \hat{Q}(\phi_{j+1, a' ; \theta^-})$  
- **There are two phases:** 
	- **Sampling**: perform actions and store the observed experiences in a replay memory
	- **Training**: select a small batch of tuples randomly and learn from this batch using gradient descent update step
		![[deep-q-learning-algo.png|600]]****
- DQL training might suffer from ***instability*** 
	- Due to combining *non-linear* Q-value function and [[Reinforcement Learning#Temporal-Difference (TD)|bootstraping]] 
	- Utilize [[#Experience Replay]], [[#Fixed Q-Target]], and [[#Double DQN]] to address instability
		![[stable-deep-q-learning-algo.png|600]]
## Experience Replay
- Used to make more *efficient* use of experiences $(S_t,A_t,R_{t+1},S_{t+1})$ during training
	- In online RL, the agent interacts with environment, gets experiences, learns from them (updates the neural network), and then discards them (**not efficient!**)
- Solution to the problem of [[#Catastrophic Forgetting]]
- Experience replay uses a **replay buffer** to save experience samples that can be reused during training
- Store experience tuples while interacting with environment and then **sample a small batch of experience tuples during training**
- Experience replay also removes correlation in observation sequences and help prevent action values from oscillating or diverging catastrophically
- ***Size of replay memory buffer is a hyperparameter that can be tuned***
## Catastrophic Forgetting
- Problem with neural networks where it **tends to forget the previous experiences as it gets new experiences**
	- e.g., if the agent is in the first level, then in the second level (which is different), the agent could forget how to behave and play in the first level
- The solution is to create a **replay buffer** and use [[#Experience Replay]] to prevent network from learning only about what it has done immediately before
## Fixed Q-Target
- When calculating TD error (aka loss), we calculate the difference between the **TD target** or **Q-target** ($R_{t+1}+\gamma \max_aQ(S_{t+1}, a)$) and the current **Q-value** (estimate of Q) ($Q(S_t, A_t)$)
	- Since *we don't have any idea of the real TD target*, we estimate it using the [[Reinforcement Learning#Bellman Equation|Bellman equation]]
	- However, if we are using the same parameters (weights) to estimate *both Q-target and Q-value* in DQL, there is *significant correlation* between the Q-target and the parameters we are changing
	- This can cause **both the Q-value and target value** shift, which is akin to chasing a moving target
	- This can lead to **significant oscillation** during training
- Solution is to use a separate network with **fixed parameters** ($\theta^-$) for estimating the TD target (Q-target)
- Copy parameters from the trained DQN every $C$ steps to update the target network
## Double DQN
- Handles the problem of **overestimation of Q-values**
- The problem originates from how we calculate the TD target in [[Reinforcement Learning#Temporal-Difference (TD)|TD]] learning
	![[TD-target-calc.png|250]]
	-  We face a simple problem: **how are we sure that the best action for the next state is the action with the ==max Q-value==?**
	- The accuracy of the Q-values depends on what we tried **and** what neighbouring states we have explored
	- Since we don't have enough information about the best action to take at the start of training, taking the maximum Q-value (which is noisy) as the best action can lead to **false positives**
	- If non-optimal actions are regularly **given higher Q-value than the optimal action, learning will be tough**
- Solution is to use two networks when we compute the Q-target (TD target)
	- **Decouple action selection from the target Q-value generation**
	- Use **original DQN network** to select best action to take for next state (i.e., action with maximum Q-value)
	- Use **Target network** to calculate the target Q-value of taking that action at the next state
- *This helps reduce overestimation and stabilize learning and train faster*
## Reference Link
- https://huggingface.co/learn/deep-rl-course/unit3/introduction 
# Policy Gradient Method
- [[Deep Learning#Neural Networks|Neural network]] used to approximate stochastic policy
	- Takes as input a state and outputs a probability distribution over actions 
	- ***Action preference***: probability of taking each action
	- Goal is to control the probability distribution of actions by tuning the network such that good actions that maximize [[Reinforcement Learning#Expected Return|expected return]] are sampled more frequently
- Difference between [[#Policy-based Method]] and Policy gradient methods
	- Policy gradient methods are a **subclass** of policy-based methods
	- Policy-based methods are *most of the time* [[Reinforcement Learning#On-policy methods|on-policy methods]]
	- Main difference between policy-based vs policy-gradient lies in **how the parameter $\theta$ is optimized**
	- **Policy-based methods** search for the optimal policy **directly**; maximize parameter $\theta$ **indirectly** by maximizing the local approximation of the *objective function* (using hill climbing, simulated annealing or evolution strategies)
	- **Policy gradient** methods search for the optimal policy **directly**; but *also optimize the parameter $\theta$* **directly** by performing gradient ascent on the performance of the objective function $J(\theta)$ 
- **Advantages**
	- **Simplicity of integration**; policy can be estimated without storing additional data (i.e. Q-values)
	- Able to learn ==**stochastic policy**== (value-based methods, like [[#Deep Q-Learning|DQL]] can't!)
		- Don't need to implement exploration-exploitation trade-off by hand
	- Get rid of problem of **perpetual aliasing**: *when two states appear (or are) the same but need different actions*
		- Via stochastic policies
	- More effective in **high-dimensional action spaces and continuous action spaces**
	- Policy gradient methods have **better convergence properties**
		- Value-based methods use *aggressive* operator by taking *maximum* over Q-estimates
		- Policy gradient methods change *smoothly over time*
- **Disadvantages**
	- Frequently *converge to local maximum* instead of global maximum
	- Can take longer to train (inefficient)
	- Have ***high variance***
- **How to optimize weights (parameters) using the expected return**
	- Idea is to let agent interact during an episode
	- If we win (get high return in) the episode, we consider each action taken was good and must be sampled in future
		- i.e., probability of taking action in state $P(a|s)$ should be increased
		- Opposite if we lost
- Stochastic policy is parameterized by $\theta$
	- Given a state, outputs a probability distribution over actions at that state
	- Denoted as,
	$$\pi_{\theta}(s) = \mathbb{P}[A|s;\theta]$$
	- $\pi_{\theta}(a_t |s_t)$ is the probability of selecting action $a_t$ from state $s_t$ given the policy
## Objective function
- How to know if our policy is *good*
	- Define an **objective function** $J(\theta)$, which gives the **performance of the agent for a trajectory (state-action sequence without rewards) and outputs the expected cumulative reward**
		$$
		\begin{align}
		J(\theta) &= E_{\tau \sim \pi}[R(\tau)] \\
			&= \sum_\tau P(\tau;\theta)R(\tau) \\
			&= \sum_\tau [\Pi_{t=0}P(s_{t+1}|s_t, a_t)\pi_{\theta}(a_t|s_t)]R(\tau)
		\end{align}
		$$
	- $R(\tau)$ is **the return** from an arbitrary trajectory $\tau$
	- $P(\tau;\theta) = \Pi_{t=0}P(s_{t+1}|s_t, a_t)\pi_{\theta}(a_t|s_t)$ is  the probability of each possible trajectory $\tau$ 
		- This probability depends on $\theta$, since it defines the policy $\pi_\theta$ that is used to select the actions of the trajectory
	- $J(\theta)$ is calculated by summing for all over all trajectories 
- **The objective is to maximize the expected return by finding the $\theta$ that will output the best action probability distributions**
		$$max_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)]$$
	- Use **gradient ascent** (inverse of [[Machine Learning#Gradient Descent|gradient descent]]) to *repeatedly update* $\theta$ (the network) in **steepest** direction of increase
		$$\theta \leftarrow \theta + \alpha * \nabla_\theta J(\theta) $$
	- Repeat above update until $\theta$ converges to the value that maximizes the objective function
	- However, there are ==**two problems**== with calculating the derivative of $J(\theta)$
		1. **Can't calculate the exact gradient of the object function** because it requires calculating the probability of each possible trajectory (very computationally expensive)
			- Rely on *sample-based (collected trajectories)* estimate to calculate a **gradient estimation**
		2. **The** **dynamics of the environment (i.e., $P(s_{t+1}|s_t, a_t)$) is needed to differentiate the objective function**, **but** ***it may not be known***
			- This can be addressed through [[#Policy Gradient Theorem]]
# Policy Gradient Theorem
- Reformulates the [[#Objective function]] of policy gradient approach into a differentiable function that **==does not require the differentiation of the state distribution (environment dynamics)==**
- For *any differentiable policy and for any policy-based objective function*, the policy [[Calculus#Gradient|gradient]] can be formulated as
	$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} [\nabla_\theta \log{\pi_\theta}(a_t|s_t)R(\tau)]$$
	 - $R(\tau)$ is **the return** from an arbitrary trajectory $\tau$
	 - The objective function is updated as, 
		 $$J(\theta) = \mathbb{E}[\log(\pi_\theta(a_t|s_t))R(\tau)]$$
## Details
- The objective function we want to optimize is 
		$$
		\begin{align}
		J(\theta) &= E_{\tau \sim \pi}[R(\tau)] \\
			&= \sum_\tau P(\tau;\theta)R(\tau) \\
			&= \sum_\tau [\Pi_{t=0}P(s_{t+1}|s_t, a_t)\pi_{\theta}(a_t|s_t)]R(\tau)
		\end{align}
		$$
	- $R(\tau)$ is the return from an arbitrary trajectory $\tau$
	- $P(\tau;\theta) = \Pi_{t=0}P(s_{t+1}|s_t, a_t)\pi_{\theta}(a_t|s_t)$ is  the probability of each possible trajectory $\tau$ 
		- This probability depends on $\theta$, since it defines the policy $\pi_\theta$ that is used to select the actions of the trajectory
	- The [[Calculus#Gradient|gradient]] for $J(\theta)$ is
		$$\nabla_\theta J(\theta) = \nabla_{\theta} \sum_\tau P(\tau;\theta)R(\tau)$$
	- We can rewrite the **gradient of the sum** as the **sum of the gradient** (because $R(\tau)$) is not dependent on $\theta$
		$$
		\begin{align}
		\nabla_\theta J(\theta) &=  \sum_\tau \nabla_{\theta} (P(\tau;\theta)R(\tau)) \\
		&= \sum_\tau \nabla_{\theta} (P(\tau;\theta))R(\tau)
		\end{align}
		$$
	- If we multiple every term by $\frac{P(\tau;\theta)}{P(\tau;\theta)}$,
		$$
		\begin{align}
		\nabla_\theta J(\theta) &= \sum_\tau \frac{P(\tau;\theta)}{P(\tau;\theta)} \nabla_{\theta} (P(\tau;\theta))R(\tau) \\
		&= \sum_\tau P(\tau;\theta) \frac{ \nabla_{\theta} P(\tau;\theta)}{P(\tau;\theta)} R(\tau) 
		\end{align}
		$$
	- Based on the calculus rule $\nabla_x logf(x)=\frac{\nabla_x f(s)}{f(x)}$, we can simplify the above to,
		$$
		\begin{align}
		\nabla_\theta J(\theta) &= \sum_\tau P(\tau;\theta) \frac{ \nabla_{\theta} P(\tau;\theta)}{P(\tau;\theta)} R(\tau)  \\
		&= \sum_\tau P(\tau;\theta) \nabla_\theta log(P(\tau;\theta)) R(\tau) \\
		&= \mathbb{E}_{\pi_\theta} [\nabla_\theta log_{\pi_\theta}(a_t|s_t)R(\tau)]
		\end{align}
		$$
	- **Some additional details** to simplify the term $\nabla_\theta log(P(\tau;\theta))$
		- We know that 
		$$
		\begin{align}
			\nabla_\theta \log(P(\tau^{(i)};\theta)) &= \nabla_\theta \log[\mu(s_0)\Pi^H_{t=0} P(s^{(i)}_{t+1}|s^{(i)}_{t}, a^{(i)}_t)\pi_{\theta}(a^{(i)}_{t}|s^{(i)}_t)] \\
			&= \nabla_\theta[\log(\mu(s_0)) + \sum^H_{t=0}\log(P(s^{(i)}_{t+1}|s^{(i)}_{t}, a^{(i)}_t)) + \sum^H_{t=0}\log(\pi_{\theta}(a^{(i)}_{t}|s^{(i)}_t))] \\
			&= \nabla_\theta \log(\mu(s_0)) + \nabla_\theta \sum^H_{t=0}\log(P(s^{(i)}_{t+1}|s^{(i)}_{t}, a^{(i)}_t)) + \nabla_\theta\sum^H_{t=0}\log(\pi_{\theta}(a^{(i)}_{t}|s^{(i)}_t))
		\end{align}
		$$
		- $\mu(s_0)$ is the **initial state distribution** 
		- $P(s^{(i)}_{t+1}|s^{(i)}_{t}, a^{(i)}_t)$ is the **dynamics of the MDP**
		- Since neither $\mu(s_0)$ or  $P(s^{(i)}_{t+1}|s^{(i)}_{t}, a^{(i)}_t)$ are dependent of $\theta$, thus the first term $\nabla_\theta log(\mu(s_0)) = 0$  and second term $\nabla_\theta \sum^H_{t=0}log(P(s^{(i)}_{t+1}|s^{(i)}_{t}, a^{(i)}_t)) = 0$
		- So we have,
			$$\nabla_\theta log(P(\tau^{(i)};\theta)) = \sum_{t=0}^H\nabla_\theta log({\pi_\theta}(a^{(i)}_t|s^{(i)}_t))$$
		
- Thus, we can estimate the [[Calculus#Gradient|gradient]] using trajectory samples using $\nabla_\theta J(\theta) \approx \widehat{g} = \frac{1}{m} \sum^m_{i=1}\sum_{t=0}^H\nabla_\theta log({\pi_\theta}(a^{(i)}_t|s^{(i)}_t))R(\tau^{(i)})$
# Reinforce Algorithm
- Also called [[Reinforcement Learning#Monte Carlo (MC)|Monte-Carlo]] policy gradient
	- It is a [[#Policy Gradient Method]] that uses an *estimated return* from an *entire episode* to update the policy parameter $\theta$
- **Main idea**
	- Loop
		1. Use policy $\pi_{\theta}$ to collect an episode $\tau$
		2. Use the episode to estimate the gradient $\nabla_\theta J(\theta) \approx \widehat{g} = \sum_{t=0}^H\nabla_\theta \log({\pi_\theta}(a_t|s_t))R(\tau)$ 
		3. Update the weights of the policy: $\theta \leftarrow \theta + \alpha \widehat{g}$  
	- $\nabla_\theta \log\pi_\theta(a_t|s_t)$ is the **direction of steepest increase of the (log) probability** of selecting $a_t$ form state $s_t$
	- $R(\tau)$ is the scoring (reward) function that **pushes probabilities for state-action pairs up** **if return is high** and **pushes probabilities down if return is low**
	- This idea can also be applied across **multiple $m$ episodes (trajectories)** to estimate the [[Calculus#Gradient|gradient]]
	- $m$ is the **batch size**
		$$\nabla_\theta J(\theta) \approx \widehat{g} = \frac{1}{m} \sum^m_{i=1}\sum_{t=0}\nabla_\theta \log({\pi_\theta}(a^{(i)}_t|s^{(i)}_t))R(\tau^{(i)})$$
- This method is **==unbiased==** because it uses the **true return** (not an estimated return) 
- Because an *entire* episode is used to calculate return, there is **high variance in policy gradient estimation**
	- Variance is due to *stochasticity* in the environment and the policy 
	- **Starting from the same state can lead to very different returns**
	![[reinforce-algo-variance-vis.png|400]]
	- Solution in Reinforce for this problem is to use **a large number of trajectories** (i.e., big batch size $m$) to reduce variance in aggregate
	- **But increasing batch size significantly reduces sample efficiency!**
# Actor-Critic Methods
- Hybrid-technique that combines [[#Value-based Method]] and [[#Policy-based Method]] to help *stabilize the training by reducing variance*
	- ==**Actor**: controls **how the agent behaves** (policy-based method)==
	- ==**Critic**: measures **how good the action taken is** (value-based method)==
## Advantage Actor-Critic (A2C)
- Solution to reduce the problem of high variance in [[#Reinforce Algorithm]]
- **Main idea**: learn two function approximations (i.e., ==**two neural networks**==)
	- A *policy* that controls how the agent acts $\pi_\theta(s)$
	- A *value function* to assists the policy update by measuring how good the action taken is $\widehat{q}_{w}(s,a)$
- **Training**
	0. At each time step $t$, we pass the current state $S_t$ from the environment to the actor and critic 
	1. The **actor** (policy) takes the state $S_t$ and outputs an action $A_t$
		![[a2c-1.png]]
	2. The **critic** (value-function) uses the state $S_t$ and action $A_t$ *(from the **actor**)* to compute the **Q-value**
		![[a2c-2.png]]
	3. The action is executed in the environment to obtain the next state $S_{t+1}$ and reward $R_{t+1}$
		![[a2c-3.png]]
	4. The actor updates its policy parameters $\theta$ using the Q-value
		$$\Delta\theta = \alpha \nabla_{\theta}(\log(\pi_{\theta}(a|s)))\widehat{q}_w(s,a)$$
		- The Q-value $\hat{q}_w$ is given by the **critic** **NN**
		- $\alpha$ is a separate learning rate for the actor network
	1. After this update, The actor produces the next action $A_{t+1}$ given the new state $S_{t+1}$
	2. The critic updates its value parameters
			$$\Delta w = \beta (R(s,a) + \gamma \widehat{q}_w(s_{t+1}, a_{t+1})- \widehat{q}_w(s_t, a_t))\nabla_w\widehat{q}_w(s_t,a_t)$$
		- $R(s,a) + \gamma \widehat{q}_w(s_{t+1}, a_{t+1})- \widehat{q}_w(s_t, a_t)$ is the [[Reinforcement Learning#TD Error|TD Error]]
		- $\beta$ is a separate learning rate for the critic network
		- $\nabla_w\widehat{q}_w(s_t,a_t)$ is the **gradient of the value function**
### Advantage Function
- Can be used in place of the [[Reinforcement Learning#Action-value function|action-value function]] to provide better stability to the model
- Also known as **==gain==**
- **Main idea**: use a function that calculates the relative of an action compared to other possible actions at a given state
	- **i.e., how taking an action at a state is better compared to the average value of the state** represented by $V(s)$
- The Advantage function is defined as,
	$$A(s,a) = Q(s,a) - V(s)$$
	- This function calculates the **extra reward we get** if we take this action at that state compared to the mean reward at the state
	- If $A(s,a) > 0$: the [[Calculus#Gradient|gradient]] pushes in that direction
	- If $A(s,a) < 0$: the [[Calculus#Gradient|gradient]] pushes in the opposite direction
- To avoid needing to implement *two value functions* $Q(s,a)$ and $V(s)$, we use the [[Reinforcement Learning#TD Error|TD error]] as an estimator 
	$$
	\begin{align}
	A(s,a) &= Q(s,a) - V(s)\\
		&\approx r + \gamma V(s')-V(s)
	\end{align}
	$$
# Proximal Policy Optimization (PPO)
- Improves agent's training stability by **preventing policy updates from being too big**
- **Builds on top of [[#Actor-Critic Methods]]**
- Use a *ratio* that indicates the **difference between the current and old policy** and **clip** this ratio to a specific range $[1-\epsilon, 1+\epsilon]$ 
	- $\epsilon$ is a tunable hyperparameter
- **Main Idea**: ==limit change you make to the policy at each training epoch== 
	- This limit is based on two reasons
		1. Smaller policy updates during training are *more likely* to converge to an optimal solution
		2. Policy updates that are too big can result in **"falling off the cliff"** (i.e., getting a bad policy), which needs a long time (or even having no possibility to recover)
	- Goal is to update the policy **conservatively**
	- To do this we need to measure how much the current policy changed compared to the previous using a *ratio calculation*
		- Clip this to a range $[1-\epsilon, 1+\epsilon]$ and remove incentive to deviate too far from the previous policy (hence **proximal** policy term)
## Surrogate Objective Function
- Update policy probabilities based on *impulse* and *advantage*
 $$L^\text{impulse}(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}A_t$$

## Clipped Surrogate Objective Function
- Redesigned [[#Objective function]] from [[#Policy Gradient Theorem]] that ==**avoids (destructively) large weight updates**==
	$$
	L^{CLIP}(\theta) = \mathbb{E}[\min(r_t(\theta)\widehat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\widehat{A}_t)]
	$$
	- $\widehat{A}_t$ is the [[#Advantage Function]]
	- $r_t(\theta)$ is the **ratio function** that estimates the *divergence* between the current policy and the old policy, defined as,
		$$
			r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} 
		$$
		- If $r_t(\theta) > 1$, the action $a_t$ is **more likely** in the current policy than the old
		- If $r_t(\theta) \in [0,1)$, then the action is **less likely**
		- The ratio $r_t(\theta)$ can replace the log probability used in the [[#Objective function]] from [[#Policy Gradient Theorem]]
		- However, without a constraint, if the action taken is much more probable in the current policy than the previous, it would lead to a *significant* policy gradient step
- The clipped part of the objective function is used to ***constrain*** the objective function
	- **Penalize changes that lead to a ratio far away from 1**
	- Ensure the policy is not too different from the previous one
	- This can be achieved through two approaches
		- **Trust Region Policy Optimization (TRPO)**: uses KL divergence constraints outside the objective function to constrain the policy update
			- This method is **complicated to implement and takes more computation time**
			- Aka PPO-penalty
		- **PPO-Clip**: clip probability ratio directly using the *clip* function
			- The ratio function $r_t(\theta)$ is limited between $[1-\epsilon, 1+\epsilon]$ 
			- **Simpler approach**
	- **Visualization of the clipping function**: 
		- ==**Note**: $p_t(\theta)$ in table is the *ratio function* (same as  $r_t(\theta)$ above)==
	![[clip-function-ppo.png|500]]
	- **For case 1 and 2**: straightforward; clip function is not used
	- **Case 3**: $p_t(\theta) < 1-\epsilon$ and $A_t > 0$ therefore,
		- $\text{min}(p_t(\theta)A_t, (1-\epsilon)A_t) = p_t(\theta)A_t$ 
		- gradient is **positive** 
		- ==We should increase action selection probabilities==
	- **Case 4**: $p_t(\theta) < 1-\epsilon$ and $A_t < 0$ therefore, 
		- $\text{min}(p_t(\theta)A_t, (1-\epsilon)A_t) = (1-\epsilon)A_t$ 
		- gradient is zero (because we don't have $\theta$ in)
		- ==We should not decrease action selection probabilities any further==
	- Same idea applies for **case 5 and 6**...
# Deep Deterministic Policy Gradient
- DDPG is an algorithm which concurrently learns a Q-function ([[Reinforcement Learning#Action-value function|action-value function]]) and a [[Reinforcement Learning#Policy|policy]] 
	- Uses [[Reinforcement Learning#Off-policy methods|off-policy]] data and [[Reinforcement Learning#Bellman Equation|Bellman Equation]] to learn the Q-function
	- Uses the Q-function to learn the policy
	- It is an [[Reinforcement Learning#Off-policy methods|off-policy]] algorithm
	- Can only bu used for environments with **continuous action spaces**
		- Can be thought of as being [[#Deep Q-Learning]] for continuous action spaces
- Closely connected to [[Reinforcement Learning#Q-learning|Q-learning]] (motivated the same way)
	- If you know the [[Reinforcement Learning#Optimal Action-value function|optimal action-value function]] $Q^*(s,a)$, then in any given state you can determine the optimal action by solving
	$$a*(s) = \arg \max_a Q^*(s,a)$$
- DDPG interleaves learning an approximator for $Q^*(s,a)$ with learning an approximator for $a^*(s)$ 
- Works for environments with ***continuous action spaces***
	- When there are a finite number of *discrete* actions, the $\max$ poses no problem; we can compute the Q-values for each action separately and directly compare them
	- But when the action space is *continuous*, we cannot exhaustively evaluate the space
	- Because action space is *continuous*, we presume $Q^*(s,a)$ is differentiable with respect to the action argument
		- This allows a gradient-based learning rule to be set up for a policy $\pi(s)$, which exploits this fact
	- Then we can approximate $\max_a Q(s,a) \approx Q(s, \pi(s))$  
- https://spinningup.openai.com/en/latest/algorithms/ddpg.html 