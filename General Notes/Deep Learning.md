#general #deeplearning #ML #neuralnetwork 
# Neural Networks
- A neural network (NN) is a mathematical function $y=f_{NN}(x)$ 
- The function $f_{NN}$ is a ***nested function***
	- A 3-layer neural network may return a scalar,
	$$y=f_{NN}(x)=f_3(\mathbf{f}_2(\mathbf{f}_1(x)))$$
	- Here, $\mathbf{f}_1$ and $\mathbf{f}_2$ are vector functions of the following form
		$$\mathbf{f}_l(\mathbf{z}) = \mathbf{g}_l(\mathbf{W}_l\mathbf{z} + \mathbf{b}_l)$$
	- $l$ is the the **layer index**, and can span from 1 to any number of layers
	- $\mathbf{g}_l$ is the **activation function**, which is a **fixed non-linear function** chosen before learning begins
	- $\mathbf{W}_l$ is a parameter matrix and $\mathbf{b}_l$ is a parameter vector for each layer, which are learned using [[Machine Learning#Gradient Descent|gradient descent]] by optimizing the cost function
		- A matrix $\mathbf{W}_l$ is used here instead of $\mathbf{w}_l$ because $\mathbf{g}_l$ is a vector function, so each row $\mathbf{w}_{l,u}$ ($u$ for unit) of the matrix $\mathbf{W}_l$ is a vector of the same dimension as $\mathbf{z}$
		- Let $a_{l,u} = \mathbf{w}_{l,u} \mathbf{z} + b_{l,u}$, the output of $\mathbf{f}_l(z)$ is  the vector $[g_l(a_{l,1}), g_l(a_{l,2})...,g_l(a_{l,\text{size}_l})]$, where $g_l$ is some scalar function and $\text{size}_l$ is the number of units in layer $l$
	- $f_3$ is a scalar function for regression task, but can also be a vector function depending on the problem
## Multi-Layer Perceptron
- Consider a **feed-forward neural network** **(FFNN)**, or **multi-layer perceptron (MLP)** show below with 2-dimensional input and 1-dimensional output
	- This FFNN can be used for regression or classification depending on the activation function used in the third (output) layer
![[multilayer-perceptron.png]]
- The NN is represented graphically as a connected combination of **units** (circles or rectangles) logically organized into one or more **layers**
	- The output of each unity is the result of the mathematical operation written inside the *rectangle units*
	- Circle units don't do anything with inputs; just pass it along
	- In each rectangle unit, all inputs are joined together to form an input vector $\mathbf{x}$
	- The unit applies a linear transformation to the input vector (same idea as [[Machine Learning#Linear Regression|linear regression]])
	- Finally, the unit applies an activation function $g$ to th result of the linear transformation and obtains the output value in $\mathbb{R}$ 
	- In *vanilla* FFNN, the output value of a unit of some layer becomes an input value of each of the units of the subsequent layer
- In the figure above, the activation function $g_l$ has one index: $l$, the index of the layer the unit belongs to
	- Usually all units of a layer use the same activation function, but this is not a rule
- Each layer can have *different number of units*
- Each unit has its own parameters $\mathbf{w}_{l,u}$ and $b_{l,u}$, where $l$ is the index of the layer and $u$ is the index of the unit
- In MLP, all outputs of one layer are connected to each input of the succeeding layer; this architecture is called **fully-connected layers**
	- The units in the layer receive as input the outputs of *each* of the units in the previous layer
## Feed-Forward Neural Network
- For regression problems, the last layer in the NN usually contains only one unit, and the activation function in the last unit is **linear**
- For binary classification, the last layer in the NN contains only one unit, and the activation function is a [[Machine Learning#Logistic Regression|logistic]] function
- The intermediate (hidden) layers' activation functions $g_{l,u}$ can be any mathematical function, so long as it's **differentiable**
	- Being differentiable is important for [[Machine Learning#Gradient Descent|gradient descent]] used to optimize the values of the parameters $\mathbf{w}_{l,u}$ and $b_{l,u}$ for all $l$ and $u$
- The activation functions should be non-linear
	- This allows the NN to approximate non-linear functions
### Activation Functions
- **Popular choices of activation functions** include: the **[[Machine Learning#Logistic Regression|logistic]] function**, **TanH**, and **ReLU**
#### **TanH** 
- Is the hyperbolic tangent function, which is similar to the logistic function but ranging from -1 to 1 (without reaching them) $$tanh(z) = \frac{e^z-e^{-z}}{e^z+e^{-z}}$$
#### **ReLU** 
- Is the rectified linear unit function, which equals zero when its input $z$ is negative and equal $z$ otherwise $$
		 relu(z) = \begin{cases}
		 0 \quad \text{if} \quad z < 0\\
		 z \quad \text{otherwise}
		 \end{cases}
		 $$
# Backpropagation
- Technique used for training neural network that is based on calculating derivatives and gradient descent
- Goal is to optimize weights so that the NN can learn how to correctly map arbitrary inputs to outputs
- [Ref](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/)
## Example
- Consider the following NN with the initial weights, bias and training inputs and outputs 
	![[backprop-1.png|300]]  
- **Forward pass**
	- Begin by determining what the NN currently predicts given the weights and biases
	- Do this by calculating the ***total net input***  to each hidden layer neuron (blue), ***squash*** the total net input using an activation function (in this case [[Machine Learning#Logistic Regression|logistic function]]), and repeat the process with the output layer
	- For example, to calculate the **total net input** for $h_1$: $$\begin{aligned} net_{h1} &= w_1i_i+w_2i_2 + b_1 \\ &= 0.15 \cdot 0.05 + 0.2 \cdot 0.1 + 0.35 = 0.3775\end{aligned}$$
	- When we **squash** it using the logistic function we get the output of $h_1$: $$out_{h_1} = \frac{1}{1+e^{-net_{h_1}}}=\frac{1}{1+e^{-0.3775}}=0.593269992$$
	- Following the same process, we obtain 
		- For $h_2$: $out_{h_2} = 0.596884378$
		- For $o_1$: $out_{o_1} = 0.75136507$
		- For $o_2$: $out_{o_2} = 0.772928465$
- **Calculate Total Error**
	- We then calculate the error for each output neuron using the squared error function and sum them to get the total error $$E_{total} = \sum \frac{1}{2}(target - output)^2$$
		- ==**Note**: the $\frac{1}{2}$ is included so that the exponent is cancelled when differentiation occurs. The result will eventually be multiplied by a learning rate, so this constant doesn't matter==
	- For example, the error for $o_1$ is $$E_{o_1} = \frac{1}{2}(target_{o_1} - out_{o_1})^2 = \frac{1}{2}(0.01 - 0.75136507)^2 = 0.274811083$$
	- Repeating this for $o_2$ we get $E_{o_2} = 0.023450026$
	- $E_{total} = E_{o_1} + E_{o_2} = 0.274811083 + 0.023450026 = 0.298371109$
- **Backwards Pass**
	- The goal with backpropagation is to update the weights such that they cause the actual output to be closer to the target output
	- In the **output layer**, consider $w_5$
		- We want to know how much a change in $w_5$ affects the total error, i.e., $\frac{\partial E_{total}}{\partial w_5}$    
		- By applying the chain rule, we obtain $$\frac{\partial E_{total}}{\partial w_5} = \frac{\partial E_{total}}{\partial out_{o_1}} \frac{\partial out_{o_1}}{\partial net_{o_1}} \frac{\partial net_{o_1}}{\partial w_5}$$
		- We can determine each term in above equation as follows^outh1
		$$
		\begin{align} 
		E_{total} &= \frac{1}{2}(target_{o_1} - out_{o_1})^2 - \frac{1}{2}(target_{o_2} - out_{o_2})^2 \\
		\frac{\partial E_{total}}{\partial out_{o_1}} &= 2 \cdot \frac{1}{2} (target_{o_1} - out_{o_1})^{(2-1)} \cdot -1 + 0\\ 
		\frac{\partial E_{total}}{\partial out_{o_1}} &= -(target_{o_1} - out_{o_1}) = -(0.01 - 0.75136507) = 0.74136507
		\end{align}
		$$
		- Next we can determine $\frac{\partial out_{o_1}}{\partial net_{o_1}}$ 
			$$
			\begin{align}
			out_{o_1} &= \frac{1}{1+e^{-net_{o_1}}} \\
			\frac{\partial out_{o_1}}{\partial net_{o_1}} &= -1\cdot (1+e^{-net_{o_1}})^{-1-1}\cdot-e^{-net_{o_1}} = \frac{e^{-net_{o_1}}}{(1+e^{-net_{o_1}})^2}=\frac{1}{1+e^{-net_{o_1}}} (\frac{e^{-net_{o_1}}}{1+e^{-net_{o_1}}}) \\
			 &= \frac{1}{1+e^{-net_{o_1}}}(\frac{e^{-net_{o_1}}+1-1}{1+e^{-net_{o_1}}}) = \frac{1}{1+e^{-net_{o_1}}}(\frac{1+e^{-net_{o_1}}-1}{1+e^{-net_{o_1}}}) \\ 
			 &= \frac{1}{1+e^{-net_{o_1}}}(1 - \frac{1}{1+e^{-net_{o_1}}}) = out_{o_1}(1-out_{o_1}) \\
			 &= 0.75136507(1-0.75136507) = 0.186815602
			\end{align}
			$$
		- Finally, we can determine $\frac{\partial net_{o_1}}{\partial w_5}$
			$$
			\begin{align}
			net_{o_1} &= w_5 \cdot out_{h_1} + w_6 \cdot out_{h_2} + b_2 \\
			\frac{\partial net_{o_1}}{\partial w_5} &=  1 \cdot out_{h_1} + 0 + 0 = 0.593269992
			\end{align}
			$$
		- Thus, putting the above all together we have,
			$$
			\begin{align}
			\frac{\partial E_{total}}{\partial w_5} &= \frac{\partial E_{total}}{\partial out_{o_1}} \frac{\partial out_{o_1}}{\partial net_{o_1}} \frac{\partial net_{o_1}}{\partial w_5} \\
			&= -(target_{o_1} - out_{o_1})\cdot out_{o_1}(1-out_{o_1}) \cdot out_{h_1}\\
				&= 0.74136507 \cdot 0.186815602 \cdot 0.593269992 = 0.082167041
			\end{align}
			$$
		- To decrease this error, we subtract this value from the current weight $w_5$ (optionally multiplied by some learning rate $\alpha = 0.5$)  $$w^+_5=w_5 - \alpha \cdot \frac{\partial E_{total}}{\partial w_5} = 0.4 - 0.5 \cdot 0.082167041 = 0.35891648$$
	- The same process is repeated for $w_6, w_7, w_8$ to obtain new weights (the weight update occurs *after* we have new weights leading into the hidden layer neurons)
	- In the **hidden layer**, consider $w_1$
		- We need to determine $$\begin{align} \frac{\partial E_{total}}{\partial w_1} &= \frac{\partial E_{total}}{\partial out_{h_1}} \cdot \frac{\partial out_{h_1}}{\partial net_{h_1}}\cdot \frac{\partial net_{h_1}}{\partial w_1} \end{align}$$
			![[backprop-2.png|400]]
		- ==The difference here is that the output of each hidden layer neuron contributes to the output of multiple output neurons==
		- So, we need to take into consideration its effect on **both** output neurons $$\frac{\partial E_{total}}{\partial out_{h_1}} = \frac{\partial E_{o_1}}{\partial out_{h_1}} +\frac{\partial E_{o_2}}{\partial out_{h_1}}$$
		- Starting with $\frac{\partial E_{o_1}}{\partial out_{h_1}}$,
			$$
			\begin{align}
			\frac{\partial E_{o_1}}{\partial out_{h_1}} &= \frac{\partial E_{o_1}}{\partial out_{o_1}} \cdot \frac{\partial out_{o_1}}{\partial net_{o_1}} \cdot \frac{\partial net_{o_1}}{\partial out_{h_1}}\\
			\end{align}
			$$
		- From earlier calculations, we know $\frac{\partial E_{o_1}}{\partial out_{h_1}} = 0.74136507$  and $\frac{\partial out_{o_1}}{\partial net_{o_1}} = 0.186815602$ and 
			$$
			\begin{align}
			net_{o_1} &= w_5 \cdot out_{h_1} + w_6 \cdot out_{h_2} + b_2 \\
			 \frac{\partial net_{o_1}}{\partial out_{h_1}} &= w_5 = 0.40
			\end{align}
			$$
		- Thus, $$\frac{\partial E_{o_1}}{\partial out_{h_1}} = 0.75136507 \cdot 0.186815602 \cdot 0.4 = 0.055399425$$
		- Following the same process, we get $\frac{\partial E_{o_2}}{\partial out_{h_1}} = -0.019049119$ 
		- Therefore, $$\frac{\partial E_{total}}{\partial out_{h_1}} = \frac{\partial E_{o_1}}{\partial out_{h_1}} +\frac{\partial E_{o_2}}{\partial out_{h_2}} = 0.055399425 + (-0.019049119) = 0.36350306$$
		- Now for $\frac{\partial out_{h_1}}{\partial net_{h_1}}$, 
			$$
			\begin{align}
			out_{h_1} &= \frac{1}{1+e^{-net_{h_1}}} \\
			\frac{\partial out_{h_1}}{\partial net_{h_1}} &= out_{h_1}(1-out_{h_1}) = 0.241300709
			\end{align}
			$$
		- And for $\frac{\partial net_{h_1}}{\partial w_1}$,
			$$
			\begin{align}
			net_{h_1} &= w_1 \cdot i_1 + w_3 \cdot i_2 + b_1 \\
			\frac{\partial net_{h_1}}{\partial w_1} &= i_1 = 0.05 
			\end{align}
			$$
		- Putting everything together, we have
			$$\begin{align} \frac{\partial E_{total}}{\partial w_1} &= \frac{\partial E_{total}}{\partial out_{h_1}} \cdot \frac{\partial out_{h_1}}{\partial net_{h_1}}\cdot \frac{\partial net_{h_1}}{\partial w_1} = 0.036350306 \cdot 0.241300709 \cdot0.05 = 0.000438568 \end{align}$$
			- This may also be written as,
				$$\begin{align} \frac{\partial E_{total}}{\partial w_1} &= \sum_o (\frac{\partial E_{o}}{\partial out_{o}}\frac{\partial out_{o}}{\partial net_{o}} \frac{\partial net_{o}}{\partial out_{h_1}}) \cdot \frac{\partial out_{h_1}}{\partial net_{h_1}}\cdot \frac{\partial net_{h_1}}{\partial w_1} \end{align}$$
	- The same process can be followed as before to update the weights $w_1, w_2, w_3, w_4$ in the hidden layer
# Deep Learning
- Deep learning refers to training NNs with more than two non-output layers (or many **hidden layers**)
- Suffers from the problem of **exploding gradient** and **vanishing gradient**
	- Exploding gradient can be addressed through **gradient clipping** and L1 or L2 regularization
	- Vanishing gradient arises during **backpropagation**, which is an algorithm for computing gradients on NNs using the chain rule
		- During gradient descent, the NN's parameters receive an update proportional to the partial derivative of the cost function with respect to the current parameter in each iteration of training
		- The problem is that in some cases, the *gradient will be vanishingly small*, which prevents some parameters form changing their value (in worse case stop the NN from further training)
		- Traditional activation functions, such as TanH have gradients in the range of $(0,1)$, and because backpropagation computes gradients by chain rule, the *effect of multiplying several of these small gradients* means that the gradient decreases exponentially as the number of layers increases
		- Modern solutions to this problem is to use better activation functions, such as ReLU, as well as techniques such as skip connections in residual NNs
## Recurrent Neural Network
- Recurrent neural networks (RNNs) are used to *label*, *classify*, or *generate* **sequences**
	- A sequence is a matrix, each row of which is a feature vector and the order of the rows matters
	- To *label* a sequence is to predict a class for each feature vector in a sequence
	- To *classify* a sequence is to predict a class for the entire sequence
	- to *generate* a sequence is to output another sequence relevant to the input sequence
- RNNs are often used in text and speech processing (because sentences are naturally sequences of words/punctuation marks)
- RNNs are **not feed-forward**; it contains **loops**
	- The idea is that each unit $u$ of the recurrent layer $l$ has a real-valued **state** $h_{l,u}$, which can be seen as the memory of the unit
	![[recurrent-network.png]]
	- 
# Transfer Learning
- In transfer learning, you use an **existing model** trained on some dataset and adapt the existing model to predict examples from another dataset (different from the one the model was built on)
	- The new dataset is not like holdout sets used for validation and testing; but rather represent some other phenomenon
	- e.g., using a model trained to label *wild animals* using a big labelled dataset to build a model to label *domestic animals*
- **Deep NNs offers advantages here and works as follows,**
	1. Build a deep NN model on the original dataset (wild animals)
	2. Compile a much smaller labelled dataset for second problem (domestic animals)
	3. Remove the last one or several layers from the first model (because these layers are usually responsible for classification or regression)
	4. Replace the removed layers with new layers adapted for new problem
	5. **"Freeze" the parameters of the earlier layers from the first model**
	6. Use Smaller labelled dataset and gradient descent to train parameters of **only the new layers**
- Even without an existing model, transfer learning can still help in situations where **labelled dataset is costly to obtain** but another (relevant) dataset is readily available
	- e.g., to build a document classification model, you have an taxonomy of labels with thousands of categories
		- Instead of paying some one to read, understand, and annotate millions of documents, you could consider using Wikipedia pages as the dataset to build the first model
		- Labels for Wikipedia pages can be obtained automatically using the category the Wikipedia page belongs to
		- Once the first model is learned to predict Wikipedia categories, you can use *transfer learning* to predict categories of the original problem using much fewer annotated examples 
# Weight Initialization
- Sets the starting point for the model's optimization process
	- When initial weights are too large, the gradients can be extremely large during backpropagation, leading to **exploding gradient** problem
	- When initial weights are too small, the gradients can vanish, making it difficult for model to learn effectively
## Common Initialization Methods
- Choice of weight initialization methods depends on architecture of the NN, activation functions used, and problem being solved
- [[#Xavier/Glorot Initialization]] and [[#MSRA Initialization]] are widely used an generally provide good results in most scenarios
### Zero Initialization
- Set all weights to zero
- Generally discouraged due to all neurons in a layer learning the same features, thus all neurons in a layer will learn the same features (symmetry breaking problems)
### Random Initialization
- Assign small random values to weights
- Can achieve this by sampling from a Gaussian distribution with a mean of 0 and small variance
### Xavier/Glorot Initialization
- Takes into account the **size of the input and output dimensions of a layer**
- Initializes weights by sampling from Gaussian distribution with mean of 0 and variance of `1/(input_size + output_size)` 
- Suitable for **sigmoid and hyperbolic tangent** activation functions
### MSRA Initialization
- Aka He Initialization
- Similar to [[#Xavier/Glorot Initialization]], but optimized for ReLU activation functions
- Use Gaussian distribution with mean of 0 and variance of `2/input_size` (variance adjusted to account for dead gradient in half of the input space (i.e., 0) for ReLU neurons)
### LeCun Initialization
- Designed for **Scaled Exponential Linear Unit (SELU)** activation functions
- Use Gaussian distribution with mean of 0 and variance of `1/input_size`
### Orthogonal Initialization
- Initializes weight matrix with an **[[Linear Algebra#Matrices#Orthogonal Matrix|orthogonal matrix]]**, preserving the orthogonality of the input features
- Useful for [[#Recurrent Neural Network]] to avoid vanishing gradients