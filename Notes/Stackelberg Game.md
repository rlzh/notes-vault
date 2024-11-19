#general #gametheory 

- sequential game where players are divided into *leaders* and *followers*
- leaders make the first move to maximise reward; followers respond accordingly to maximise their own

### Stackelberg Equilibrium
- Consider a two-player [[Game Theory#Normal-form Game|normal-form]], [[Game Theory#General-sum Game|general-sum]] game, where there is a leader $l$ and a follower $f$ 
$$
A_l \; \text{is the finite set of actions the leader has}
$$
$$
A_f \; \text{is the finite set of actions the follower has}
$$
$$ 
\Delta^l \; \text{is the set of probability distributions over the leader's actions}
$$
$$ 
\Delta^f \; \text{is the set of probability distributions over the follower's actions}
$$
$$
u_i(a_l, a_f) \; \text{for} \; i \in {l, f} \; \text{i the utilities for the players}
$$
with slight abuse in notation,
$$
u_i(x,y) = \mathbb{E}_{a_l \sim x,a_f \sim y}[u_i(a_l, a_f)]
$$
where $x\in \Delta^l$, $y \in \Delta^f$ are probability distributions over $A_l$ and $A_f$ 
- In general, we assume leader commits to a strategy $x \in X$, and given such an $x$, the follower chooses their strategy from their best-response set,
$$
BR(x) = \arg \max_{y\in \Delta^f} u_f(x,y)
$$
- The goal of the leader is to chose a strategy $x$ maximising their utility subject to the follower's best response,
$$
\max_{x\in \Delta^l} u_l(x,y) \; s.t. \; y \in BR(x)
$$
- **There is a problem with the above optimisation problem; $BR(x)$ may be *set valued* and $u_l(x,y)$ would generally differ depending on which $y \in BR(x)$ is chosen**
#### Strong Stackelberg Equilibrium (SSE)
- Assume the follower breaks ties in favour of the leader
- In this case, the optimisation problem becomes,
$$
\max_{x\in \Delta^l,y\in BR(x)} u_l(x,y)
$$
- This is the most *optimistic* variant
- **Always guaranteed to exists** 
#### Weak Stackelberg Equilibrium (WSE)
- Assume follower breaks ties in adversarial manner
- This is the most *pessimistic* variant

$$
\max_{x\in \Delta^l} \min_{y \in BR(x)} u_l(x,y)
$$
- **Not always guaranteed to exists**
