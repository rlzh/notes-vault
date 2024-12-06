
#general  #stats
# Event
- A set of outcomes from an experiment (subset of the sample space) to which a probability is assigned
- A single outcome may be an element of many different events
- An event consisting of only a single outcome is called an *elementary event (or atomic event)*

# Random Variable
- A mathematical formalization of a quantity or object which depends on random events
- Often refers to a mathematical function
## Domain
- The set of possible outcomes in a sample space
## Range
- The measurable space 
# Latent Variable
- A variable that cannot be directly observed but is instead estimated from other observed variables
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