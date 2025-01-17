
#general  #stats #math 
# Event
- A set of outcomes from an experiment (subset of the sample space) to which a probability is assigned
- A single outcome may be an element of many different events
- An event consisting of only a single outcome is called an *elementary event (or atomic event)*
# Random Variable
- A mathematical formalization of a quantity or object which depends on random events
- Often refers to a mathematical function
## Discrete Random Variable
- A discrete random variable takes on only countable number of distinct values (e.g., red, yellow, blue)
- The **probability distribution** of a discrete random variable is described by a list of probabilities associated with each of its possible values
	- This list of probabilities is called a **probability mass function (pmf)**
	- Each probability in a pmf is greater than or equal to 0; and the sum of probabilities is equal to 1
## Continuous Random Variable
- A continuous random variable takes on infinite number of possible values in some interval (e.g., height, weight, time)
- Because the number of values of a continuous random variable $X$ is infinite, the probability $Pr(X=c)$ for any $c$ is 0
- The probability distribution of a continuous random variable is described by a **probability density function (pdf)** 
	- The pdf is a function whose co-domain is non-negative and the area under the curve is equal to 1
## Domain
- The set of possible outcomes in a sample space
## Range
- The measurable space 
- aka co-domain
# Latent Variable
- A variable that cannot be directly observed but is instead estimated from other observed variables
# Expected value
- The expected value is a generalization of the weighted average
- Informally, it is the mean of the possible values a [[#Random Variable]] can take, weighted by the probability of those outcomes
- For ***discrete*** random variable (with finite number of outcomes), the expected value is a weight sum of all possible outcomes
	- Let a discrete random variable $X$ have $k$ possible values $\{x_i\}^k_{i=1}$ 
	- The expectation of $X$ denoted as $\mathbb{E}[X]$ is given by,
		$$\mathbb{E}[X] = \sum^k_{i=1}[x_i\cdot Pr(X=x_i)]$$
	- $Pr(X=x_i)$ is the probability that X has the value $x_i$ based on the pmf
- For ***continuous*** random variables (with a continuum of possible outcomes), the expected value is defined by *integration*
	$$\mathbb{E}[X] = \int_{\mathbb{R}}xf_X(x)dx$$
	- $f_X$ is the pdf of the variable $X$ and $\int_{\mathbb{R}}$  is the integral of the function $xf_X$ 
	- Note: the property of pdf implies the area under the curve is 1 and $\int_\mathbb{R}f_X(x)dx=1$ 

# Standard Deviation
- Is defined as 
	$$\sigma = \sqrt{\mathbb{E}[(X-\mu)^2]}$$
- For discrete random variable, the standard deviation is given by,
	$$\sigma = \sqrt{Pr(X=x_1)(x_1 - \mu)^2+ Pr(X=x_2)(x_2 - \mu)^2 + ... + Pr(X=x_k)(x_k-\mu)^2} $$
# Unbiased Estimators
- When pdf $f_X$ is unknown, but we have a sample $S_X=\{x_i\}^N_{i=1}$, we often content ourselves with the unbiased estimators (instead of the true values of statistics of the probability distribution, such as the expectation)
- $\hat{\theta}(S_X)$ is an unbiased estimator of some statistic $\theta$ calculated using a sample $S_X$ drawn from an unknown probability distribution if $\hat{\theta}(S_X)$ has the following property:
	$$\mathbb{E}[\hat{\theta}(S_X)] = \theta$$
	- where $\hat{\theta}$ is a **sample statistic** obtained using a sample $S_X$ and not the real statistic $\theta$ that can be obtained only knowing $X$ (when the expectation is taken over all possible samples drawn from $X$)
	- This means if you have an *unlimited* number of samples $S_X$, and you compute some unbiased estimator, such as $\hat{\mu}$, using each sample, then the average of all these $\hat{\mu}$ equals the real statistic $\mu$ that you would get computed on $X$
	- It can be shown that an unbiased estimator of an unknown $E[X]$ (expected value) is given by $\frac{1}{N}\sum^{N}_{i=1}x_i$  (This is called the **sample mean**)
# Parameter Estimation
- Deals with when we have a model of a random variable $X$'s distribution, say $f_\theta$ 
	- This model is a function that has some parameters in the form of a vector $\theta$ 
	- e.g., Gaussian function that has two parameters, $\mu$ and $\sigma$, and is defined as 
	$$f_{\theta}(s) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$
	- where $\theta = [\mu,\sigma]$ and and $\pi$ is the constant 3.14159...
	- We can update the values of the parameters in the vector $\theta$ from sample data using [[Bayes' Theorem]]
	$$Pr(\theta = \hat{\theta}|X=x) \leftarrow \frac{Pr(X=x|\theta=\hat{\theta})Pr(\theta=\hat{\theta})}{Pr(X=x)} = \frac{Pr(X=x|\theta=\hat{\theta})Pr(\theta=\hat{\theta})}{\sum_{\tilde{\theta}}Pr(X=x|\theta=\tilde\theta)Pr(\theta=\tilde\theta)}$$^bayes
	- where $Pr(X=x|\theta=\hat\theta) \overset{\text{def}}{=} f_{\hat\theta}$ 
- If we have a sample $\mathcal{S}$ of $X$ and the set of possible values for $\theta$ is *finite*, we can easily estimate $Pr(\theta=\hat\theta)$ by applying [[Bayes' Theorem]] iteratively, one example $x \in \mathcal{S}$ at a time
	- The initial value $Pr(\theta = \hat\theta)$ (i.e., **prior**) can be guessed such that $\sum_{\hat\theta} P(\theta=\hat\theta) = 1$ 
	- We first compute $Pr(\theta=\hat\theta|X=x_1)$ for all possible values of $\hat\theta$ 
	- Then before updating $Pr(\theta = \hat\theta | X = x)$ again using $x=x_2$, we replace the prior $Pr(\theta=\hat\theta)$ in eq.[[#^bayes]] by the new estimate $Pr(\theta=\hat\theta) \leftarrow \frac{1}{N} \sum_{x\in\mathcal{S}} Pr(\theta=\hat\theta|X=x)$ 
	- The best value of the parameters $\theta^*$ given one example is obtained using the principle of **maximum a posteriori** 
		$$\theta^* = \text{arg}\max_{\theta} \Pi^N_{i=1} Pr(\theta=\hat\theta|X=x_i)$$
		- If the set of possible values for $\theta$ is not finite, then we need to optimize equation above using a numerical optimization routine, such as [[Machine Learning#Gradient Descent|gradient descent]]
		- Usually we optimize the logarithm of the above right-hand side expression (since multiplication turns into sum in logarithms)
# Marginal Probability
- Probability of an event irrespective of other random variables
- Denoted as $P(A)$
# Joint Probability
- Probability of two or more simultaneous events 
- Denoted as $P(A\cap B)$ or $P(A \; \text{and} \; B)$ 
# Conditional Probability
- Probability of one (or more) events given the occurrence of another event
- Denoted as $P(A|B)$ 
## Definition
$$
P(A|B) = \frac{P(A\cap B)}{P(B)}, \text{if} \; P(B) \neq 0
$$
# Predictions

## Null Hypothesis
- Claim that the effect being studied *does not exist*
- The test of significance is designed to assess the strength of the evidence against the null hypothesis
	- Hypothesis that no relationship between two sets of data or variables being analyzed
	- **If null hypothesis is true, any experimentally observed effect is due to chance alone (hence "null")**
	- A statement of "no effect" or "no difference"
	- **Example**: Given the test scores of two random samples, one of men and one of women, does the one group score better than the other?
		- A possible null hypothesis is that the mean male score is the same as the mean female score
		$$
		H_0: \mu_1 = \mu_2
		$$
		where, 
		$$
		\begin{gather}
		H_0 = \text{null hypothesis} \\
		\mu_0 = \text{mean of male scores} \\
		\mu_1 = \text{mean of female scores}
		\end{gather}
		$$
		- A stronger null hypothesis is that the two samples have equal variance and shapes for their respective distributions

|                 | Actual Yes | Actual No |
| --------------- | ---------- | --------- |
| **Predict Yes** | TP         | FP        |
| **Predict No**  | FN         | TN        |

## Recall
- aka true positive rate (TPR) 
$$
Recall = \frac{TP}{TP+FN}
$$

## Precision 
$$
Precision = \frac{TP}{TP+FP}
$$

## Accuracy
$$
Accuracy = \frac{TP+TN}{TP+TN+FP+FN}
$$
## Specificity
$$
Specificity = \frac{TN}{TN+FP}
$$

## False Positive Rate
- Type I Error
- Rejection of null hypothesis when it is actually true
$$
FP \; Rate = \frac{FP}{TN+FP}
$$
## False Negative Rate
- Type II Error
- Failure to reject null hypothesis when actually false
$$
FN \; Rate = \frac{FN}{TN+FP}
$$
## F1-score
- Harmonic mean of precision and recall
$$
F1 = \frac{2\cdot Precision \cdot Recall}{Precsision + Recall}
$$
## ROC Curve
- Receiver operating characteristic (ROC) (curve comes from radar engineering) 
- Commonly used method to assess the performance of classification models
- Uses a combination of [[#Recall]] and [[#False Positive Rate]] to build up a summary picture of classification performance
- ROC curves can only be used to asses classifiers that return some confidence score (or a probability) of prediction
- To draw a ROC curve, you first discretize the range of confidence score
	- e.g., if the range is $[0,1]$, then you can discretize into $[0,0.1,0.2,0.3...,0.9,1]$
- Then, use each discrete value as the **prediction threshold** and predict the labels of examples in dataset using the model and the threshold
	- If the threshold is 0, all predictions will be positive, so both TPR and and FPR will be 1
	- If the threshold is 1, all predictions will be negative, so both TPR and FPR will be 0
- The higher the **area under the ROC curve (AUC)** the better the classifier
	- an AUC higher than 0.5 is better than a random classifier
	- a perfect AUC is 1
	- Usually a good classifier can be obtained from a well-behaved model by selecting the threshold value that gives TPR close to 1 and FPR near 0
# Kullback-Leibler Divergence 
- KL divergence is a type of **statistical "distance"**: ==a measure of how much a model probability distribution $Q$ is different from a *true* probability ditribution $P$==
	- It is mathematically defined as,
		$$D_{KL}(P||Q) = \sum_{x\in\mathcal{X}} P(x)log(\frac{P(x)}{Q(x)})$$
- Simple interpretation of KL divergence of $P$ from $Q$ is the **[[#Expected value|expected]]** excess *surprise* from using $Q$ as a model instead of $P$ when the actual distribution is $P$
- Not a real "distance" (not a metric)
	- Does not satisfy the triangle inequality
- Also called **relative entropy and I-divergence** 