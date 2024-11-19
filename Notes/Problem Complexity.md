#general

[ref](https://stackoverflow.com/questions/1857244/what-are-the-differences-between-np-np-complete-and-np-hard)

**Decision Problem:** A problem with a **yes** or **no** answer

(listed in increasing difficulty...)
# P (Polynomial)
- complexity class that represents the set of all decision problems that can be solved in polynomial time w.r.t to input size
- given an instance of the problem, the answer can be decided in polynomial time
- e.g., Given a connected graph $G$, can its vertices be coloured using two colours so that no edge is monochromatic?

# NP (Non-deterministic Polynomial)

- complexity class that represents set of all decision problems for which instances where the answer is *yes* have proofs that can be verified in polynomial time
- given an instance of the problem and a certificate (or witness) to the answer being *yes*, we can check that it is correct in polynomial time
- e.g., integer factorisation, where we are given $m$ and $n$ and we need to find an integer *f* with $1 < f < m$, such that $f$ divides $n$

# NP-complete
- complexity class that represents set of all problems $X$ in NP for which it is possible to reduce any other NP problem $Y$ into $X$ in polynomial time
- this means we can solve $Y$ quickly if we know how solve $X$ quickly
- $Y$ is reducible to $X$ if there is a polynomial time algorithm $f$ to transform instances $y \in Y$ to instances of $x=f(y) \in X$ in polynomial time, with property that answers to $y$ is yes iff the answer to $f(y)$ is yes
- e.g., 3-SAT ...
- this is an important class of problems because if a deterministic polynomial time algorithm can be found to solve one of them, *then every NP problem is solvable in polynomial time*

# NP-hard
- complexity class that represents set of problems *at least as hard as the NP-complete problems*
- NP-hard problems do not have to be in NP, and they do not have to be decision problems
- a problem $X$ is NP-hard, if there is an NP-complete problem $Y$, such that $Y$ is reducible to $X$ in polynomial time 
- since any NP-complete problem can be reduced to any other NP-complete problem in polynomial time, all NP-complete problems can be reduced to any NP-hard problem in polynomial time; so, if there is a solution to one NP-hard problem in polynomial time, there is a solution to all NP problems in polynomial time
- e.g., halting problem ...

### Relationship between P, NP, NP-complete, and NP-hard

![[Pasted image 20241117212410.png]]