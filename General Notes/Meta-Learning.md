#general #meta-learning #ML 
- Goal of meta-learning is to design a ML model that can learn new concepts and skills fast with a few training examples
	- **Learning to learn**
	- Well-defined for any family of ML problems: supervised learning, [[Reinforcement Learning]], etc
		- e.g., A classifier trained on non-cat images can tell whether a given image contains a cat after seeing a handful of cat pictures, a game bot is able to quickly master a new game, mini-robot completes test
- [Ref](https://lilianweng.github.io/posts/2018-11-30-meta-learning)
# Problem (supervised learning)
- A good meta-learning model should be trained over a variety of learning tasks and optimized for the best performance on a distribution of tasks (including potentially unseen tasks)
	- Each task is associated with a dataset $\mathcal{D}$, containing both *feature vectors and true labels*
	- The optimal model parameters are $$\theta^* = \arg \min_\theta \mathbb{E}_{\mathcal{D} \sim p(\mathcal{D})}[\mathcal{L}_\theta(\mathcal{D})]$$
- **Few-shot classification** is a instantiation of meta-learning in field of supervised learning
	- The **dataset $\mathcal{D}$ is often split into 2 parts**
		- A **support set** $S$ for learning
		- A **prediction set** $B$ for training or testing
		- $\mathcal{D} = \textlangle {S, B} \textrangle$ 
		- Classification tasks considered in format *$K$-shot $N$-class classification task* (support set contains $K$ labelled examples for each of $N$ classes)
	- A dataset $\mathcal{D}$ contains pairs of feature vectors and labels, $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}$
	- Each label belongs to a known label set $\mathcal{L}^{\text{label}}$ 
	- The classifier $f_\theta$ with parameter $\theta$ outputs a probability of a data point belonging to the class $y$ give the feature vector $\mathbf{x}$, $P_\theta(y|\mathbf{x})$ 
	- The optimal parameters should maximize the probability of true labels across multiple training batches $B \subset \mathcal{D}$ $$\theta^* = \arg \max_\theta \mathbb{E}_{B\subset \mathcal{D}}[\sum_{(\mathbf{x},y)\in B}P_\theta(y|\mathbf{x})]$$
	- In few-shot classification, ==the goal is to reduce the prediction error on data samples with ***unknown*** **labels** given a small support set for **fast learning**==
	- To mimic what happens during inference, we try to "fake" datasets with a *subset* of labels to avoid exposing all labels to the model and modify the optimization procedure accordingly to encourage fast learning
		1. Sample a subset of labels $L \subset \mathcal{L}^{\text{label}}$	
		2. Sample a support set $S^L \subset \mathcal{D}$ and a prediction set $B^l \subset \mathcal{D}$. 
			- Both of them only contain data points with labels from the sampled label set $L$ (i.e., $y \in L, \forall(x,y) \in S^l, B^L$)
		3. The support set $S^L$ is part of the model input
		4. The final optimization uses the prediction set $B^L$ to compute the loss and update the model parameters through [[Deep Learning#Backpropagation|backpropagation]]
	-  One may consider each pair of sampled dataset $(S^L,  B^L)$ to be a single data point
		- The model is trained to generalized on other datasets and achieve the following learning objective $$\theta = \arg \max_\theta \mathbb{E}_{L\subset\mathcal{L}}[\mathbb{E}_{S^L \subset \mathcal{D}, B^L \subset \mathcal{D}}[\sum_{(x,y)\in B^L}P_{\theta}(x,y,S^L)]]$$
# Common Approaches
## Optimization-based
- [[Deep Learning]] models learn through [[Deep Learning#Backpropagation|backpropagation]] of [[Calculus#Gradient|gradients]], but gradient-based optimization is not inherently designed to work with a small number of training samples, nor converge within a small number of optimization steps
- The goal of optimization-based approaches for meta-learning is to adjust the optimization algorithm so that the model can become good at learning with few examples
### Model-Agnostic Meta-Learning (MAML)
- Paper: [[2017-PMLR-model-agnostic-meta-learning-for-fast-adaptation-of-DN-Finn.pdf]]
- General optimization algorithm that is compatible with any model that learns through [[Machine Learning#Gradient Descent|gradient descent]]
- Give a model $f_\theta$ with parameters $\theta$, a task $\tau_i$, and its associated dataset $(\mathcal{D}^{(i)}_\text{train}, \mathcal{D}^{(i)}_\text{test})$ 
	- We can update the model parameters by **one or more gradient descent steps** as follows (in this case only 1 task is sampled==?==)$$\theta'_i = \theta - \alpha \nabla_\theta\mathcal{L}_{\tau_i}^{(0)}(\theta)$$
		- Where $\mathcal{L}^{(0)}$ is the loss computed using the mini-batch with id $(0)$ 
	- The above formula only optimizes for *one task*; to achieve good generalization across a variety of tasks, we would like to find the optimal $\theta^*$ so that task-specific fine-tuning is more efficient
	- So, we sample a new data batch with id $(1)$ for updating the **meta-objective**
		- The loss for mini-batch $\mathcal{L}^{(1)}$ only depends on mini-batch $(1)$
		$$\theta^* = \arg \min_\theta \sum_{\tau_i \sim p(\tau)} \mathcal{L}_{\tau_i}^{(i)}({\theta'_{i}}) = \arg \min_\theta \sum_{\tau_i \sim p(\tau)} \mathcal{L}_{\tau_i}^{(i)}({\theta - \alpha \nabla_\theta\mathcal{L}_{\tau_i}^{(0)}(\theta)})$$
		$$\theta \leftarrow \theta - \beta \nabla_\theta \sum_{\tau_i \sim p(\tau)} \mathcal{L}^{(1)}_{\tau_i}({\theta - \alpha \nabla_\theta\mathcal{L}_{\tau_i}^{(0)}(\theta)})$$
![[maml.png|300]]
- Thick line is the meta-training process
- $\theta$ is the parameter vector being meta-learned
- $\nabla \mathcal{L}_i$ direction of gradient step with respect to task $i$
- $\theta_i^*$ is the optimal parameter vector for task $i$ 
	![[maml-algo.png|300]]
#### First-Order MAML
- The meta-optimization step above relies on calculating second derivatives
- FOMAML makes implementation simpler by omitting second order derivatives
- Consider the case of performing $k$ inner gradient steps, $k \geq 1$, starting with initial model $\theta_\text{meta}$  (**inner loop**)
	$$
	\begin{aligned}
	\theta_0 &= \theta_{\text{meta}} \\
	\theta_1 &= \theta_0 - \alpha \nabla_\theta\mathcal{L}_{\tau_i}^{(0)}({\theta_0}) \\
	\theta_2 &= \theta_1 - \alpha \nabla_\theta\mathcal{L}_{\tau_i}^{(0)}({\theta_1}) \\ 
	... \\
	\theta_k &= \theta_{k-1} - \alpha \nabla_\theta\mathcal{L}_{\tau_i}^{(0)}({\theta_{k-1 }})
	\end{aligned}
	$$
- Then,  the **outer loop** samples a new data batch for updating the meta-objective
	$$\theta_\text{meta} \leftarrow \theta_\text{meta} - \beta g_\text{MAML}$$
	- where 
	$$
	\begin{aligned}
	g_\text{MAML} &= \nabla_\theta \mathcal{L}^{(1)}({\theta_k}) \\
		&= \nabla_{\theta_k} \mathcal{L}^{(1)}({\theta_k})\cdot (\nabla_{\theta_{k-1}}{\theta_k}) ... (\nabla_{\theta_{0}}{\theta_1})(\nabla_{\theta}{\theta_0})\\
		&= \nabla_{\theta_k} \mathcal{L}^{(1)}({\theta_k}) \cdot (\Pi_{i=1}^{k} \nabla_{\theta_{i-1}}\theta_i)\cdot I\\ 
		&= \nabla_{\theta_k} \mathcal{L}^{(1)}({\theta_k}) \cdot (\Pi_{i=1}^{k}\nabla_{\theta_{i-1}}(\theta_{i-1}-\alpha\nabla_\theta \mathcal{L}^{(0)}(\theta_{i-1}))) \\
		&= \nabla_{\theta_k} \mathcal{L}^{(1)}({\theta_k}) \cdot (\Pi_{i=1}^{k} I - \alpha \nabla_{\theta_{i-1}}(\nabla_\theta \mathcal{L}^{(0)}(\theta_{i-1})))
	\end{aligned}
	$$
- FOMAML ignores the second derivative $\nabla_{\theta_{i-1}}(\nabla_\theta \mathcal{L}^{(0)}(\theta_{i-1}))$ and simplifies into the following $$g_\text{FOMAML} = \nabla_{\theta_k} \mathcal{L}^{(1)}(\theta_{k})$$
### Reptile
- Paper: [[2018-arxiv-first-order-meta-learning-algo-Nichol.pdf]]
- Simple meta-learning algorithm (similar to [[#Model-Agnostic Meta-Learning (MAML)]] in many ways) 
- Rely on meta-optimization through gradient descent and is model-agnostic