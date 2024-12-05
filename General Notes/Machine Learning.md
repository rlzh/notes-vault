
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
