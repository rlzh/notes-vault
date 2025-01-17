#general #math 
# Gradient 
- The **gradient** of a *continuous function* $f$ is defined as the vector that contains the partial derivatives $\frac{\delta f}{\delta x_i}(p)$ computed at a point $p = (x_1, ...,x_n)$ in $n$-dimensional space
	- ==**The gradient gives the direction and rate of fastest increase**==
		- If the gradient at point $p$ is non-zero, 
			- Then the direction of the gradient is the *direction in which the function increase the most quickly* from $p$ 
			- The magnitude of the gradient is the *rate of increase* in that direction
	- The gradient is finite and defined *if and only if* all partial derivatives are also defined and finite
	- Formal notation indicate the gradient as
	$$\nabla f(p) = [\frac{\delta f}{\delta x_1}(p), ...,\frac{\delta f}{\delta x_n}(p),]^T$$
- Given a point on the function, the direction of the gradient is the normal direction of the tangential hyperplane ([[https://www.reddit.com/r/math/comments/5vq89h/why_is_the_gradient_normal_to_the_tangent_plane/|ref]])