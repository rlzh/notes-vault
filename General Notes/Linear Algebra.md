#general  #math #linearalgebra
# Unit Vector
- A vector with a length of 1
- Aka normalized vector
# Orthonormality
- Two vectors in an inner (dot) product space are orthonormal if they are orthogonal [[#Unit Vector|unit vectors]]
# Dot Product
- Aka inner product
- Suppose $\textbf{x} = (x_1, x_2)$ and $\textbf{y} = (y_1, y_2)$ are two vectors in $\mathbb{R}^2$, neither of which is the zero vector
- Let $\alpha$ and $\beta$ be the angles between $\textbf{x}$ and $\textbf{y}$ and the positive horizontal axis, measured in the counterclockwise direction
- Suppose $\alpha \geq \beta$ and $\theta = \alpha - \beta$, then $\theta$ is the angle between $\textbf{x}$ and $\textbf{y}$ measured in the counterclockwise direction 
	![[angle-between-vectors.png|200]]
- The subtraction formula cosine gives,
		$$\cos(\theta) = \cos(\alpha-\beta) = \cos(\alpha)\cos(\beta) + \sin(\alpha)\sin(\beta)$$
	- Now, 
	$$
	\begin{gather}
	\cos{(\alpha)} = \frac{x_1}{||\mathbf{x}||}, \\
	\cos{(\beta)} = \frac{y_1}{||\mathbf{y}||}, \\
	\sin{(\alpha)} = \frac{x_2}{||\mathbf{x}||}, \\
	\sin{(\beta)} = \frac{y_2}{||\mathbf{y}||}, \\
	\end{gather}
	$$
	- Thus, 
	$$
	\cos{(\theta)} = \frac{x_1 y_1}{||\mathbf{x}||||\mathbf{y}||} + \frac{x_2 y_2}{||\mathbf{x}||||\mathbf{y}||} = \frac{x_1 y_1 + x_2 y_2}{||\mathbf{x}||||\mathbf{y}||}
	= \frac{\mathbf{x} \cdot \mathbf{y}}{{||\mathbf{x}||||\mathbf{y}||}}$$
	- Or, 
	$$\mathbf{x}\cdot \mathbf{y} = ||\mathbf{x}||||\mathbf{y}|| \cos{(\theta)}$$
## Properties
- For any vectors $\mathbf{x}$, $\mathbf{y}$, and $\mathbf{z}$ in $\mathbb{R}^n$ and scalar $\alpha$,
	$$
	\begin{gather}
		\mathbf{x} \cdot \mathbf{y} = \mathbf{y} \cdot \mathbf{x}, \\
		\mathbf{x} \cdot (\mathbf{y}+\mathbf{z}) = \mathbf{x} \cdot \mathbf{y} + \mathbf{x} \cdot \mathbf{z}, \\
		(\alpha \mathbf{x}) \cdot \mathbf{y} = \alpha (\mathbf{x} \cdot \mathbf{y}), \\
		0 \cdot \mathbf{x} = 0 \\,
		\mathbf{x} \cdot \mathbf{x} \geq 0,
		\mathbf{x} \cdot \mathbf{x} = 0 \quad \text{only if} \quad \mathbf{x} = \mathbf{0}, \\
		\mathbf{x} \cdot \mathbf{x} = ||\mathbf{x}||^2
	\end{gather}
	$$
- Vectors $\mathbf{x}$ and $\mathbf{y}$ in $\mathbf{R}^n$ are said to be **orthogonal** (or **perpendicular**) $\mathbf{x} \perp \mathbf{y}$ if $\mathbf{x} \cdot \mathbf{y} = 0$
# Cross Product
- Only defined in $\mathbb{R}^3$
- Given vectors $\mathbf{a}$ and $\mathbf{b}$ in $\mathbb{R}^3$
	$$
	\mathbf{a} \times\mathbf{b} = \begin{bmatrix}
								a_1 \\
								a_2 \\
								a_3 \\
								\end{bmatrix} \times 
								\begin{bmatrix}
								b_1 \\
								b_2 \\
								b_3 \\
								\end{bmatrix}
								= \begin{bmatrix}
								a_2b_3 - a_3b_2 \\
								a_3b_1 - a_1b_3 \\
								a_1b_2 - a_2b_1 \\
								\end{bmatrix}
								
	$$
	- The resulting vector is **orthogonal** to $\mathbf{a}$ and $\mathbf{b}$
## Properties
- $||\mathbf{a} \times \mathbf{b}|| = ||\mathbf{a}||||\mathbf{b}||\sin{(\theta)}$
	- **Proof**
		$$
		\begin{align} 
		||\mathbf{a}\times \mathbf{b}||^2 &= (a_2b_3-a_3b_2)^2 + (a_3b_1-a_1b_3)^2 + (a_1b_2-a_2b_1)^2 \\
			&= a_2^2b_3^2 - 2a_2a_3b_2b_3 + a_3^2b_2^2 + a_3^2b_1^2 - 2a_1a_3b_1b_3 + a_1^2b_3^2 + a_1^2b_2^2 - 2a_1a_2b_1b_2 + a_2^2b_1^2 \\
			&= a_1^2(b_2^2 + b_3^2) + a_2^2(b_1^2+b_3^2) + a_3^2(b_1^2+b_2^2)-2(a_2a_3b_2b_3 + a_1a_3b_1b_3 + a_1a_2b_1b_2)
		\end{align}
		$$
		$$
		||\mathbf{a}||^2||\mathbf{b}||^2\cos^2{(\theta)} = a_1^2b_1^2 + a_2^2b_2^2 + a_3^2b_3^2 + 2(a_2a_3b_2b_3 + a_1a_3b_1b_3 + a_1a_2b_1b_2)
		$$
		$$
		\begin{align}
		||\mathbf{a} \times \mathbf{b}||^2 + ||\mathbf{a}||^2||\mathbf{b}||^2\cos^2{(\theta)} &= a_1^2(b_1^2+b_2^2 + b_3^2) + a_2^2(b_1^2+b_2^2+b_3^2) + a_3^2(b_1^2+b_2^2+b_3^2) \\
		&= (b_1^2+b_2^2+b_3^2) (a_1^2+a_2^2+a_3^2) \\ 
		&= ||\mathbf{b}||^2 ||\mathbf{a}||^2 \\
		||\mathbf{a} \times \mathbf{b}||^2 &= ||\mathbf{a}||^2||\mathbf{b}||^2  -  ||\mathbf{a}||^2||\mathbf{b}||^2\cos^2{(\theta)} \\
		 ||\mathbf{a} \times \mathbf{b}||^2 &= ||\mathbf{a}||^2||\mathbf{b}||^2 (1 - \cos^2{(\theta)}), \quad \text{recall:}\quad \sin^2\theta + \cos^2\theta = 1 \\
		 ||\mathbf{a} \times \mathbf{b}||^2 &= ||\mathbf{a}||^2||\mathbf{b}||^2 \sin^2{(\theta)}
		\end{align}
		$$

# Cauchy-Schwarz Inequality
- For all $\mathbf{x}$ and $\mathbf{y}$ in $\mathbb{R}^n$,
	$$|\mathbf{x} \cdot \mathbf{y}| \leq ||\mathbf{x}|| ||\mathbf{y}||$$
	- $|\mathbf{x}\cdot\mathbf{y}| = ||\mathbf{x}||||\mathbf{y}||$  only when $\mathbf{x} = c\mathbf{y}$, where $c$ is some scalar value
- **Proof**
	- Assume $\mathbf{x}$ and $\mathbf{y}$ are fixed vectors in $\mathbb{R}^n$, with $\mathbf{y} \neq \mathbf{0}$, let $t$ be a real number and consider the following function
	$$
	\begin{align}
	f(t) &= ||t\mathbf{y} - \mathbf{x} ||^2 \geq 0 \\
		&= (t\mathbf{y} - \mathbf{x} ) \cdot (t\mathbf{y}-\mathbf{x}) \\
	     &= t\mathbf{y} \cdot t\mathbf{y} - \mathbf{x} \cdot t \mathbf{y} - t\mathbf{y} \cdot \mathbf{x} + \mathbf{x} \cdot \mathbf{x}\\
	     &=  ||\mathbf{y}||^2t^2 - 2(\mathbf{x}\cdot\mathbf{y}) t + ||\mathbf{x}||^2
	\end{align}
	$$

	- Now $f(t) \geq 0$ should hold for all values of $t \in \mathbb{R}$; thus, let $t=\frac{2(\mathbf{x}\cdot\mathbf{y})}{2||\mathbf{y}||^2}$ 
	- Then we have,
	$$
	\begin{align}
	f(\frac{2(\mathbf{x}\cdot\mathbf{y})}{2||\mathbf{y}||^2}) &= ||\mathbf{y}||^2(\frac{2(\mathbf{x}\cdot\mathbf{y})}{2||\mathbf{y}||^2})^2 - 2(\mathbf{x}\cdot{\mathbf{y}})\frac{2(\mathbf{x}\cdot\mathbf{y})}{2||\mathbf{y}||^2} + ||\mathbf{x}||^2 \geq 0 \\
	&= \frac{(\mathbf{x}\cdot\mathbf{y})^2}{||\mathbf{y}||^2} - \frac{2(\mathbf{x}\cdot\mathbf{y})^2}{||\mathbf{y}||^2} + ||\mathbf{x}||^2  \geq 0 \\
	&= \frac{(\mathbf{x}\cdot\mathbf{y})^2 - 2(\mathbf{x}\cdot\mathbf{y})^2}{||\mathbf{y}||^2} + ||\mathbf{x}||^2 \geq 0 \\
	&= ||\mathbf{x}||^2 \geq \frac{(\mathbf{x}\cdot\mathbf{y})^2}{||\mathbf{y}||^2} \\
	&= ||\mathbf{x}||^2 ||\mathbf{y}||^2 \geq (\mathbf{x} \cdot \mathbf{y})^2 \\
	\end{align}
	$$
	- Taking the positive square root of this gives the inequality
# Triangle Inequality
- If $\mathbf{x}$ and $\mathbf{y}$ are vectors in $\mathbb{R}^n$, then
	$$
		||\mathbf{x} + \mathbf{y} || \leq ||\mathbf{x}|| + ||\mathbf{y}||
	$$
- **Proof**
		$$
		\begin{align}
		||\mathbf{x} + \mathbf{y}||^2 &= (\mathbf{x} + \mathbf{y}) \cdot (\mathbf{x} + \mathbf{y}) \\
		&= \mathbf{x} \cdot \mathbf{x} + 2(\mathbf{x}\cdot\mathbf{y}) + \mathbf{y}\cdot\mathbf{y} \\
		&= ||\mathbf{x}||^2 + 2(\mathbf{x}\cdot\mathbf{y}) + ||\mathbf{y}||^2  
		\end{align}
		$$
	- Given the [[#Cauchy-Schwarz Inequality]], it follows that
		$$
		||\mathbf{x} + \mathbf{y}||^2 \leq ||\mathbf{x}||^2 + 2||\mathbf{x}||||\mathbf{y}|| + ||\mathbf{y}||^2 = (||\mathbf{x}|| + ||\mathbf{y}||)^2
		$$
	- Taking the square root of the above gives the inequality	
# Line
- If $\mathbf{v}$ is a non-zero vector in $\mathbb{R}^n$, then for any scalar $t$, $t\mathbf{v}$ has the same direction as $\mathbf{v}$ when $t > 0$ and opposite direction when $t < 0$
- The set of points $\{t\mathbf{v}: -\infty < t < \infty\}$ forms a line through the origin
- If we add a vector $\mathbf{p}$ to each of these points, we obtain a **line** or the set of all points in $\{t\mathbf{v}+\mathbf{p}: -\infty < t < \infty\}$ which is a line through $\mathbf{p}$ in the direction of $\mathbf{v}$
![[line.png|200]]
# Hyperplane
- Two vectors $\mathbf{x}$ and $\mathbf{y}$ in $\mathbf{R}^n$ are said to be **linearly independent** if neither one is a scalar multiple of the other
	- Geometrically, $\mathbf{x}$ and $\mathbf{y}$ are linearly independent if they do not lie on the same line through the origin
- A hyperplane is a set 
	 $$\{x:a^Tx = b\}$$
	 - where $a \in \mathbb{R}^n$ and $b \in \mathbb{R}$ 
	 - $a$ defines the normal vector of the hyperplane
	 - $b$ is the offset


# Matrices

## Invertible Matrix
- An invertible matrix is a **square** matrix which has an inverse
	- If some other matrix is multiplied by the invertible matrix, the result can be multiplied by an inverse matrix to undo the operation
- An $n$-by-$n$ square matrix $A$ is called invertible (aka **nonsingular, nondegenerate**) if there exists an $n$-by-$n$ matrix $B$ such that  $$AB = BA = I_n$$
	- where $I_n$ denotes the $n$-by-$n$ identity matrix
- In this case, the matrix $B$ is uniquely determined by $A$ and is called the **inverse** of A, denoted by $A^{-1}$
## Orthogonal Matrix
- Is a real square matrix whose columns and rows are [[#Orthonormality|orthonormal]] vectors
- One way to express this is, $$Q^TQ = QQ^T == I$$
	 - where $Q^T$  is the transpose of Q and I is the identify matrix
 - This leads to the equivalent characterization: a matrix $Q$ is orthogonal if its transpose is equal to its [[#Matrices#Invertible Matrix|inverse]]: $$Q^T = Q^{-1}$$ 