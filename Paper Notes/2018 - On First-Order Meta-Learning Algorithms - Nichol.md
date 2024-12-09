#paper #RL 
**author:** Nichol
**conference/journal:** arXiv
**year**: 2018
**file:** [[2018-arxiv-first-order-meta-learning-algo-Nichol.pdf]]
# Summary
- Propose approach to learn a parameter initialization that can be fine-tuned quickly on a new task using only first-order derivatives for the meta-learning updates
- Bayesian inference can be used to explain why human learning abilities is more efficient (data-wise) than machine learning algorithms
	- But making algorithms more Bayesian is challenging due to computational requirements (intractable)
- Solution-based on **Model-agnostic Meta-Learning (MAML)**

> Rather than trying to emulate Bayesian inference (which may be computationally intractable), meta-learning seeks to directly optimize a fast-learning algorithm, using a dataset of tasks
[[2018-arxiv-first-order-meta-learning-algo-Nichol.pdf#page=1&selection=53,0,54,92|ref]]

# Approach
## MAML
- Consider the optimization problem of MAML
	- find an initial set of parameters $\phi$ such that for a randomly sampled task $\tau$ with corresponding loss $L_t$, the learner will have low loss after $k$ updates
	$$minimize_{\phi} \mathbb{E}_{\tau}[L_\tau(U^k_\tau(\phi))]$$
		- $U^k_\tau$ is operator that updates $\phi$ $k$ times using data sampled from task $\tau$
		- In few-shot learning, $U$ corresponds to performing [[Machine Learning#Gradient Descent|gradient descent]] or [[Machine Learning#Gradient Descent#Adam|Adam]] on batches of data sampled from $\tau$ 
	- MAML solves a version of the above equation that makes an additional assumption: 
		- For a given task $\tau$, the inner-loop optimization uses training samples $A$, whereas the loss is computed using test samples $B$ (omitting the superscript $k$)
		$$minimize_{\phi} \mathbb{E}_{\tau}[L_{\tau,B}(U_{\tau,A}(\phi))]$$
	- MAML works by optimizing the above loss through [[Machine Learning#Gradient Descent#Stochastic Gradient Descent|stochastic gradient descent]], computing
		$$
		\begin{align}
		g_{MAML} &= \frac{d}{d\phi}  L_{\tau,B}(U_{\tau,A}(\phi)) \\
			&= U'_{\tau,A}(\tau)L'_{\tau,B}(\tilde{\phi}) \\
			&= U'_{\tau,A}(\tau)L'_{\tau,B}(U_{\tau,A}(\phi))
		\end{align}
		$$
		- $U'_{\tau, A}(\phi)$ is the Jacobian matrix of the update operation $U_{\tau, A}$
		- $U_{\tau, A}$ corresponds to adding a sequence of gradient vectors to the initial vector
			- $U_{\tau, A} (\phi)= \phi + g_1 + g_2 + ... + g_k$ 
## FOMAML
- First-order MAML
- Treats the Jacobian $U'_{\tau, A}$  as the identity operation (i.e., multiply by one)
- The gradient used by FOMAML in the outer-loop is
	$$g_{FOMAML} = L'_{\tau,B}(U_{\tau,A}(\phi))$$
	- Thus, FOMAML can be implemented in a simpler way

## Reptile
![[reptile-algo.png]]

# Evaluation

# Implementation
todo...
# Notables
todo...
# Future Work
todo...


