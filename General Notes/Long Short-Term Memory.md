#general #deeplearning #ML 

- A type of [[Deep Learning#Recurrent Neural Network|recurrent neural network]]
# Problem of long-term dependencies
- The main appeal of RNNs is that they might be able to connect previous information to present task
	- Sometimes, we only need to look at recent information 
		- e.g., predicting the last word in "the clouds are in the *sky*"
		- In these cases, we don't need any further context
		- The gap between the relevant information and the place that it's needed is **small**
		- RNN can learn to use past information
	- Sometimes, we need more context
		- e.g., predicting the last word in the text "I grew up in France... I speak fluent *French*"
		- Recent information only suggests the next word is probably the name of a language, but we need to narrow down which language (we need the context of **France** from further back)
		- ==**As the gap grows, RNNs become unable to learn to connect the information==**
-  In theory, RNNs should be able to handle *long-term dependencies*
	- In practice, RNNs don't seem to be able to learn them (e.g., due to the problem of [[Machine Learning#Vanishing Gradient|vanishing gradient]])
	- **LSTM networks are designed to solve this problem**
# Core Idea
- LSTM networks are a special kind of RNN capable of learning long-term dependencies
	- Remembering information for long periods of time is their *default behaviour*
- All RNNs have form of chain of repeating modules of neural network
	- In standard RNNs, the repeating module will have very simple structure (e.g., a single [[Deep Learning#TanH|tanh]] layer)
	![[rnn-module.png|400]]
- LSTMs  have similar chain structure, but the repeating module has a different structure
	- There a four neural network layers 
	![[lstm-module.png|400]]
- The key to LSTMs is the **cell state**, horizontal line running through top of the diagram
	- The cell state is like a *conveyor belt*
	- Runs straight down the entire chain, with only some minor linear interactions
	- Allow for information to easily flow along it unchanged
	![[lstm-cell-state.png|300]]
- LSTMs have the ability to remove or add information to the cell state, which are regulated by structures called **gates**
	- Gates are a way to optionally let information through
	- Gates are composed of a [[Machine Learning#Logistic Regression|sigmoid]] layer and a point-wise multiplication operation
	![[lstm-gate.png]]
	- The sigmoid layer outputs values in the range $(0,1)$, which represents how much of each component should be let through
- An LSTM has three such gates to protect and control the cell state
## Details
- The first step of the LSTM is to decide what information to throw away from the cell state
	- This decision is made by a sigmoid layer called the **forget gate layer**
	- The forget gate layer looks at $h_{t-1}$ and $x_t$ and outputs a value in the range of $(0,1)$  for each number in the cell state $C_{t-1}$, to indicate whether to keep the information or get rid of it
		![[lstm-forget-gate.png|500]]
- The next state is to decide what new information we are going to store in the cell state
	- First, a [[Machine Learning#Logistic Regression|sigmoid]] layer called the **input gate layer** decides which values to update
	- Then, the [[Deep Learning#TanH|tanh]] layer creates a vector of new candidate values $\hat{C}_t$ that could be added to the cell state
	![[lstm-step-2.png|500]]
- Lastly, we need to decide what to output
	- **The output is based on the cell state**
	- First, we run a sigmoid layer, which decides what parts of the cell state to output
	- Then, the cell state is put through tanh (to push values between $-1$ and $1$) and multiply it with the output of the sigmoid layer to only output the parts chosen 
	![[lstm-step-3.png|500]]
# Variants on LSTM
## Peephole Connections
- Variation on standard LSTM that allow gate layers to look at cell state
	- Optionally add peepholes to gates (doesn't have to be added to all gates)
	![[lstm-peephole.png|500]]
## Coupled Forget and Input Gates
- Make decisions regarding forget and input together
	- Only forget when we are going to input something in its place
	![[lstm-coupled.png|500]]
## Gated Recurrent Unit (GRU)
- Combine forget and input gates into a single **update gate**
- Merges cell state and hidden state
![[lstm-gru.png|500]]