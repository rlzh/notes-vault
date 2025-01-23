#general #math 
- Algorithm used for optimization problems with *large search spaces*
- A metaheuristic optimization technique (by Kirkpatrick et al.)  to solve Travelling Salesman problem
- Based on annealing process used in metallurgy, where metal is heated to high temperature quickly and cooled slowly to allow material to be more ductile and easier to work with
- In SA, a search process starts in a high-energy state (initial solution)
	- Gradually lowers the *temperature* (control parameter) until it reaches a minimum energy state (optimal solution)
- Main advantage
	- Ability to escape from local minima and converge to a global minimum
	- Easy to implement
	- No need for prior knowledge of search space
# General Idea 
- Start with initial solution
- Iteratively improve current solution by randomly **perturbing** it and accepting the perturbation with a certain probability
	- **Perturbation should not deviate too far from current solution**
	- The probability of accepting a worse solution is initially high. but gradually decreases as the number of iterations increases
	- Accuracy depends on number of iterations SA performs