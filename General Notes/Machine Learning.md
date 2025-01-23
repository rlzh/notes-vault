#general #ML 
# Hyperparameters
- A hyperparameter is a property of a learning algorithm (usually having a numerical value)
- The value of the hyperparameter influences the way the algorithm works, but the hyperparameters are not learned by the algorithm itself from data
- Hyperparameters need to be configured before running the algorithm
# Parameter
- Parameters are variables that define the model learned by the learning algorithm
- Parameters are directly modified by the learning algorithm based on the training data

# Fundamental Algorithms
## Linear Regression
- learns a model which is a *linear combination* of features of the input example
- **Problem**
	- Have a collection of labelled examples $\{(x_i, y_i)\}^N_{i=1}$, where $N$ is the size of the collection and $x_i$ is the $D$-dimensional feature vector of example $i=1,...,N$, $y_i$ is a real-valued target and every feature $x_i^{(j)}, j=1,...,D$ is also a real number
	- Want to build a model (or parameterized function) $f_{{w},b}(s)$ as a linear combination of features of example $x$ 
		$$f_{w,b}(x) = wx + b$$
		- where $w$ is a $D$-dimensional vector of parameters and $b$ is a real number
	- Use the model to predict unknown $y$ for a given $x$: $y \leftarrow f_{w,b}(x)$ 
	- Want to find the *optimal parameter values* $(w^*, b^*)$
	![[linear-regression.png|300]]
- **Solution**
	- To find optimal parameter values $w^*$ and $b^*$, **minimize** the following expression, also know as the **mean squared error (MSE)**
	$$\frac{1}{N} \sum_{i=1...N} (f_{w,b}(x_i) - y_i)^2$$
- Linear regression is simple, which means it is not likely to *overfit*
## Logistic Regression
- Is not a regression algorithm (but a **classification learning algorithm**)
	- Name comes from statistics due to the mathematical formulation of logistic regression is similar to linear regression
- **Problem**
	- In **binary logistic regression**, still want to model $y_i$ as a function of $x_i$, however **binary** $y_i$ means this is not straightforward
		- The linear combination of features, such as $wx_i+b$ is a function that spans from $-\infty$ to $\infty$, while $y_i$ only has two possible values
		- This can be generalized to multi-class classification
		- A function that can address this problem is the **standard logistic function or sigmoid function**, where a value closer to 0 is assigned a negative label and a value closer to 1 is assigned a positive label, given by,
		$$ f(x) = \frac{1}{1+e^{-x}} $$
		![[sigmoid-func.png|300]]
		- The **logistic regression model** is defined as,
		$$ f_{w,b}(x) = \frac{1}{1+e^{-(wx+b)}}$$^lrm
- **Solution**
	- To find the optimal parameter values $w^*$ and $b^*$, we maximize the **likelihood** of our training set according to the model
		- Likelihood function defines how likely the observation (an example) is according to our model
		- e.g., given labelled observation $(x_i, y_i)$ in our training data, assume we found/guessed some specific values $\hat{w}$ and $\hat{b}$ of our parameters. 
			- If we apply our model $f_{\hat{w},\hat{b}}$ to $x_i$ using eq. [[#^lrm]], we will get some value $0 < p < 1$ as output
			- If $y_i$ is the positive class, the *likelihood* of $y_i$ being the positive class according to our model is given by $p$
			- If $y_i$ is the negative class, the *likelihood* of $y_i$ being the negative class is given by $1-p$
	- The *optimization criterion* in logistic regression is called **maximum likelihood**, such that we now maximize the likelihood of the training data according to our model
		$$L_{w,b} = \Pi_{i=1...N} f_{w,b}(x_i)^{y_i}(1-f_{w,b})^{(1-y_i)}$$
	- In practice, it is more convenient to ***maximize*** **the** **log-likelihood** instead,
		$$LogL_{w,b} = ln(L_{w,b}(x)) = \sum^N_{i=1}[y_i \ln{f_{w,b}(x)} + (1-y_i)\ln(1-f_{w,b}(x))]$$
		- This avoids numerical overflow
		- Because $\ln$ is a *strictly increasing function*, maximizing this function is the same as maximizing its argument; solution to this function is the same as the original problem
## SVM
- Support Vector Machine (SVM) is a supervised learning algorithm that requires positive labels with numeric value of $+1$ and negative labels with the value of $-1$
- The algorithm puts all feature vectors on an imaginary $D$-dimensional plot and draws an imaginary $(D-1)$-dimensional line (i.e., a *hyperplane*) that separates examples with positive labels from examples with negative labels.
	- This boundary separating the examples of different classes is called the **decision boundary**
- The equation for the hyperplane is given by two parameters
	- A real-valued vector $\mathbf{w}$ of the same dimensionality as our input feature vector $\mathbf{x}$, and a real number $b$ such that,		
	$$\mathbf{wx} - b = 0$$
- The predicted label for some input feature vector $x$ is given by,
	$$y=\text{sign}(\mathbf{wx}-b)$$
	- where $\text{sign}$ is a mathematical operator that takes any value as input and returns $+1$ if the input is positive or $-1$ if the input is negative
- The goal of the algorithm is to leverage the dataset and find the optimal values $\mathbf{w^*}$ and $\mathbf{b^*}$ 
	- To find the optimal values $\mathbf{w^*}$ and $\mathbf{b^*}$, we solve an *optimization problem* with constraints
	- The constraints include
		- Given training examples in pairs $(\mathbf{x_i}, y_i)$, we want
			$$
			\begin{gather}
			\mathbf{wx_i} - b \geq + 1 \quad \text{if} \quad y_i = +1,\\
			\mathbf{wx_i} - b \leq - 1 \quad \text{if} \quad y_i = -1 
			\end{gather}
			$$
		- We also prefer is the hyperplane separates positive examples from negative ones with the largest **margin**
			- Margin is the distance between the closest examples of two classes, as defined by the decision boundary
			- This is achieved by *minimizing* the Euclidean norm of $\mathbf{w}$, denoted $||\mathbf{w}||$ and given by $\sqrt{\sum^{D}_{j=1} (\mathbf{w}^{(j)})^2}$ 
			- Minimizing the Euclidean norm of $\mathbf{w}$ (i.e., $||\mathbf{w}||$) provides the highest margin between two classes because equations $\mathbf{wx}-b=1$ and $\mathbf{wx}-b=-1$ define two *parallel hyperplanes*, and the distance between these hyperplanes is given by $\frac{2}{||\mathbf{w}||}$, so smaller $||\mathbf{w}||$ means larger distance between the hyperplanes 
			- **Proof**
				- let $\mathbf{x_1}$ be any point in the first hyperplane $\mathbf{wx} = 1+b$ and consider the line $L$ that passes through $\mathbf{x_1}$ in the direction of the vector $\mathbf{w}$; the equation for $L$ is given by $\mathbf{x_1} + \mathbf{w}t$ for $t \in \mathbb{R}$ 
				- The intersection of $L$ and the second hyperplane $\mathbf{wx}= -1+b$
					$$\mathbf{w}(\mathbf{x_1}+\mathbf{w}t) = -1+b$$
				- Thus, 
				$$
				\begin{align}
				t &= \frac{((-1+b)-\mathbf{wx_1})}{\mathbf{w}\mathbf{w}} \\
				 &= \frac{(-1+b)-(1+b)}{\mathbf{ww}}\\
				 &= \frac{-2}{\mathbf{ww}} \\
				 &= \frac{-2}{||\mathbf{w}||^2}
				\end{align}
				$$
				- Therefore the intersection point is $\mathbf{x_2} = \mathbf{x_1} + \mathbf{w} \frac{-2}{||\mathbf{w}||^2}$ 
				- The distance between the two points $\mathbf{x_1}$ and $\mathbf{x_2}$ is the distance between the hyperplanes
					$$
					||\mathbf{x_1 - x_2}|| = ||\mathbf{x_1} - (\mathbf{x_1}+\mathbf{w}\frac{-2}{||\mathbf{w}||^2})|| = ||\mathbf{w}||\frac{|-2|}{||\mathbf{w}||^2} = \frac{2}{||\mathbf{w}||}
					$$
				
	- Thus, the final optimization problem is given by,
	$$
	\begin{gather}
	\min ||w|| \\
	s.t. \quad y_i(wx_i-b) \geq 1 \quad \text{for} \quad i = 1,...,N
	\end{gather}
	$$
![[svm.png|400]]	
# Decision Trees (DT)
- A DT is an ***acyclic* graph** that can be mused to make decisions
	- In each **branching node** of the graph, a specific feature $j$ of the feature vector is examined
	- If the value of the feature is below a specific threshold, then the left branch is followed; otherwise, the right branch is followed
	- Decisions are made at the **leaf nodes** about which class to which example belongs
- Requires little data preparation (don't require feature scaling or centering)
- DTs are sensitive to small variations in data (unstable!)
- DTs are **non-parametric model** (vs [[#Linear Regression|linear regression]], [[#Logistic Regression|logistic regression]] - parametric model)
- DTs have *more degree of freedom*, doesn't make assumption about data (doesn't assume linear)
- **Problem Definition**
	- Give a collection of labelled examples (labels belong to the set $\{0,1\}$), we want to build a DT that would allow us to predict the class given a feature vector
- **Solution**
	- There are various formulations to DT learning algorithms
	- Consider **ID3** here, where the optimization criterion is the average log-likelihood (which we aim to **maximize**)
	$$ \frac{1}{N} \sum^N_{i=1}[y_i \ln{f_{ID3}(x)} + (1-y_i)\ln(1-f_{ID3}(x))]$$
		- where $f_{ID3}$ is a decision tree
	- Contrary to [[#Logistic Regression|logistic regression]], ID3 algorithm optimizes the criterion approximately by constructing a **non-parametric model** $f_{ID3}(x) = Pr(y=1|x)$
	- The ID3 learning algorithm works as follows
		- Let $\mathcal{S}$ denote a set of labelled examples $\{(x_i,y_i)\}^N_{i=1}$
		- To begin, the DT only has a start node that contains all examples
		- Start with a constant model defined as,
			$$f^{\mathcal{S}}_{ID3} = \frac{1}{|S|}\sum_{(x,y)\in\mathcal{S}}y$$^constant-model
		- The prediction given by $f^{\mathcal{S}}_{ID3}$ would be the same for any input $x$
		- Then, we search through all features $j=1,...,D$ and all thresholds $t$ and split the set $\mathcal{S}$ into two subsets $\mathcal{S}_- = \{(x,y)|(x,y)\in\mathcal{S}, x^{(j)} < t\}$ and $\mathcal{S}_+ = \{(x,y)|(x,y)\in\mathcal{S}, x^{(j)} \geq t\}$ 
		- The two subsets would go to two leaf nodes, and we evaluate, for all possible pairs $(j,t)$ how good the split $\mathcal{S}_-$ and $\mathcal{S}_+$ is
		- Finally, we pick the best such values $(j^*, t^*)$, split $\mathcal{S}$ into $\mathcal{S}_-$ and $\mathcal{S}_+$, form two new leaf nodes and continue recursively on $\mathcal{S}_-$ and $\mathcal{S}_+$ (or quit if no split produces a model that's *sufficiently better* than the current one)
	- "goodness" of a split is estimated based on **entropy**
		- **Entropy** is a measure of uncertainty about a random variable
		- Entropy reaches its maximum when all values of the random variable are *equally probable* 
		- Entropy reaches its minimum when the random variable can have only one value
		- Entropy of a set of observations $\mathcal{S}$ is given by,
			$$ H(\mathcal{S}) = -f_{ID3}^{\mathcal{S}}\ln{f_{ID3}^{\mathcal{S}}} - (1-f_{ID3}^{\mathcal{S}})\ln{(1-f_{ID3}^{\mathcal{S}})}$$
		- When a set of observations is split by a certain feature $j$ and a threshold $t$, the **entropy of a split** is a weighted sum of two entropies,
		$$H(\mathcal{S}_-,\mathcal{S}_+) = \frac{|\mathcal{S}_-|}{|\mathcal{S}|}H(\mathcal{S}_-) + \frac{|\mathcal{S}_+|}{|\mathcal{S}|}H(\mathcal{S}_+)$$^ent-split
		- In ID3, at each step, at each leaf node, we find the split that minimizes the entropy given by eq. [[#^ent-split]] or we stop at this leaf node
	- The algorithm *stops* at a leaf node in any of the below conditions
		- All examples in the leaf node are classified correctly by the constant model ([[#^constant-model]])
		- We cannot find an attribute to split upon
		- The split reduces entropy by less than some $\epsilon$ 
		- The tree reaches some maximum depth $d$
	- The algorithm doesn't guarantee an optimal solution (decisions to split on each iteration is local)
	- Most widely used formulation of DT learning algorithm is called **C4.5**, which is an extension on ID3
## Gini Impurity
- Score to evaluate if all training instances it applies to belong to same class (how well this decision boundary separates classes). 0 is best = only 1 class under this branch
    - Gini score = 1 - sum{ ratio of class instance among training instances in this node }^2
- ***Entropy***: (Info Theory) $H = -\sum p_i \cdot log(p_i)$, can be used in place of Gini (similar results)
## CART training algorithm 
- Pick single feature and threshold to create split until maximum depth is reached. minimize cost function
    - cost function = $left\% \cdot left Gini score + right\% \cdot right Gini score$ (greedy algorithm, only tries to optimize current level not in the long run) finding optimal tree is [[Problem Complexity#NP-complete|NP-complete]]
- Logarithmic complexity
## Regularization
- max depth, min samples before split, min samples per leaf, etc to regularize (prevent over-fit of) DT
## Regression
-  try to split data in way that minimizes cost function based on MSE
    - cost function = $left\% \cdot MSE_{left} + right\% \cdot MSE_{right}$
    - $MSE =\sum (\text{average of samples} - y_i)^2$
# Gradient Descent
- **Iterative optimization algorithm that's used when training a machine learning model**
	- Based on a *convex* function and tweaks its parameters iteratively to minimize a given cost/loss function
- Requires a direction and a learning rate
	- **Learning rate** $\alpha$: determines the size of the steps that are taken to reach the minimum
	- **Cost/loss function**: measures the difference between *actual value* and *predicted value* 
		- Provide feedback to the model so that it can adjust its parameters 
- The idea is to **continuously move along the** **negative [[Calculus#Gradient|gradient]] (steepest descent)** until the cost function is close to zero
![[gradient-descent.png|300]]
- The [[Calculus#Gradient|gradient]] is calculated with respect to a vector of parameters for the model, typically the weights $w$
	- In neural networks, the process of applying gradient descent is called **backpropagation of error**
		- Backpropagation uses the sign of the gradient to determine whether the weights should **increase or decrease**
			$$w_{n+1} = w_{n} - \alpha \nabla_w f(w)$$
## Local Minima and Saddle Points
- For convex problems, gradient descent can find the global minimum with ease
- For non-convex problems, gradient descent can struggle to find the global minimum
	- The slope of the cost function can be at zero or close to zero at ***local minima and saddle points***
- Local minima mimic the shape of a global minimum
- [[#Stochastic Gradient Descent|Noisy gradients]] can help with escaping local minima and saddle points
## Vanishing Gradient
- Occurs when [[Calculus#Gradient|gradient]] is too small as we move backwards during back propagation
- This causes the earlier layers of the neural network to learn more slowly than later layers
	- Weight parameter updates become insignificant; algorithm no longer learning
## Exploding Gradient
- When the [[Calculus#Gradient|gradient]] is too large, which creates an *unstable* model
	- Model weights grows too large and eventually be represented as NaN
## Batch Gradient Descent
- Sums the error for each point in a training set and updates the model only after all training examples have been evaluated
	- This process is called a ***training epoch***
- Provides computation efficiency but can have long processing time for large training datasets
	- Needs to store all of the data into memory
- Usually produces stable error gradient
## Stochastic Gradient Descent
- Runs a training epoch for each example within the dataset and updates each training example's parameter one at a time
	- Results in **noisy gradients** due to frequent update
- Easier to store in memory (since only need to hold one training example)
- Trade-off speed for computational efficiency
## Mini-batch Gradient Descent
- Combines [[#Batch Gradient Descent]] and [[#Stochastic Gradient Descent]]
- Splits dataset into small batch sizes and performs updates on each batch
![[gradient-descent-visual.png|500]]
## Adam
- Optimization algorithm that can be used instead of classical [[#Stochastic Gradient Descent]], which maintains a single learning rate for all weight updates 
- Adam computes individual adaptive learning rates for different parameters from estimates of the first and second moments of the gradient
- Combine the advantages of the two extensions of [[#Stochastic Gradient Descent]]
	- **Adaptive Gradient Algorithm** (AdaGrad): maintains a per-parameter learning rate to improve performance on sparse gradients
	- **Root Mean Square Propagation** (RMSProp): maintains per-parameter learning rate based on average of recent magnitudes of gradients for each weight
- Adam calculates the exponential moving average of gradient and the squared gradient
	- Two parameters $\beta_1$ and $\beta_2$ control the decay rates of the moving averages
# One-hot
- A grouping of bits among which the legal combination of values are those with a single high (i.e., 1) bit and all the others are low (i.e., 0)
	- e.g. $[0,0,0,1,0,0]$ 