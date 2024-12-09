
# Decision Trees (DT)
- Requires little data preparation (don't require feature scaling or centering)
- DTs are sensitive to small variations in data (unstable!)
- DTs are non-parametric model (vs Linear regression - parametric model)
- DTs have more degree of freedom, doesn't make assumption about data (doesn't assume linear)
## Gini impurity
- Score to evaluate if all training instances it applies to belong to same class (how well this decision boundary separates classes). 0 is best = only 1 class under this branch
    - Gini score = 1 - sum{ ratio of class instance among training instances in this node }^2
- Entropy: (Info Theory) $H = -\sum p_i \cdot log(p_i)$, can be used in place of Gini (similar results)
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
- Optimization algorithm that's used when training a machine learning model
	- Based on a *convex* function and tweaks its parameters iteratively to minimize a given cost/loss function
- Requires a direction and a learning rate
	- **Learning rate**: determines the size of the steps that are taken to reach the minimum
	- **Cost/loss function**: measures the difference between *actual value* and *predicted value* 
		- Provide feedback to the model so that it can adjust its parameters 
- The idea is to continuously move along the negative gradient (steepest descent) until the cost function is close to zero
## Local Minima and Saddle Points
- For convex problems, gradient descent can find the global minimum with ease
- For non-convex problems, gradient descent can struggle to find the global minimum
	- The slope of the cost function can be at zero or close to zero at ***local minima and saddle points***
- Local minima mimic the shape of a global minimum
- [[#Stochastic Gradient Descent|Noisy gradients]] can help with escaping local minima and saddle points
## Vanishing Gradient
- Occurs when gradient is too small as we move backwards during back propagation
- This causes the earlier layers of the neural network to learn more slowly than later layers
	- Weight parameter updates become insignificant; algorithm no longer learning
## Exploding Gradient
- When the gradient is too large, which creates an *unstable* model
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

## Adam
- Optimization algorithm that can be used instead of classical [[#Stochastic Gradient Descent]], which maintains a single learning rate for all weight updates 
- Adam computes individual adaptive learning rates for different parameters from estimates of the first and second moments of the gradient
- Combine the advantages of the two extensions of [[#Stochastic Gradient Descent]]
	- **Adaptive Gradient Algorithm** (AdaGrad): maintains a per-parameter learning rate to improve performance on sparse gradients
	- **Root Mean Square Propagation** (RMSProp): maintains per-parameter learning rate based on average of recent magnitudes of gradients for each weight
- Adam calculates the exponential moving average of gradient and the squared gradient
	- Two parameters $\beta_1$ and $\beta_2$ control the decay rates of the moving averages