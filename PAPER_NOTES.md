# PAPER_NOTES: Learning representations by back-propagating errors

## 1. The Central Claim
**What is the paper asserting works better, and why?**
The paper claims that their new "back-propagation" learning procedure allows multi-layer neural networks to learn complex internal representations of data, a feat that earlier methods (like the perceptron-convergence procedure) could not achieve. 

Earlier models failed because they didn't know how to adjust the weights of "hidden" layers that weren't directly connected to the output.The authors assert that by using the chain rule from calculus to propagate the error backwards from the output layer, they can accurately calculate exactly how much "blame" to assign to every single weight in the hidden layers.This allows the network to systematically adjust all its weights via gradient descent to minimize total error.

## 2. The Core Architecture & Algorithm
**What exactly needs to be implemented to test that claim?**
To verify this, we need to implement a feed-forward, layered neural network with the following strict components:
* **Architecture:** An input layer, at least one hidden layer, and an output layer.
* **Forward Pass:** Each node computes a total input ($x_j$) as a linear weighted sum of the outputs from the layer below.This sum is then passed through a non-linear activation function (the sigmoid function: $1 / (1 + e^{-x})$) to produce the node's final output.
* **Loss Function:** The total error is calculated using the Sum of Squared Differences between the actual output and the desired output.
* **Backward Pass (Optimization):** The weights must be updated using gradient descent.The update rule must also include an "acceleration method" (momentum) where the current weight change is influenced by the previous weight change to prevent getting stuck in local minima.

## 3. Dataset, Metrics, and Baseline
* **Dataset:** The paper demonstrates this architecture on tasks that strictly require hidden features, such as detecting mirror symmetry in a 1D binary array. For this implementation, we will generate a symmetrical/non-symmetrical binary array dataset exactly as described in the paper.
* **Evaluation Metric:** The paper evaluates success based on the reduction of the total accumulated error ($E$).Specifically, a prediction is considered correct if an output unit that should be "on" has an activation $> 0.8$, and a unit that should be "off" has an activation $< 0.2$.
* **Baseline:** The implicit baseline is the standard Perceptron , which mathematically fails at these non-linear tasks (like symmetry or XOR) because it cannot train hidden units.