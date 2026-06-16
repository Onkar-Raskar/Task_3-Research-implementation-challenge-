# 1986 Symmetry Detection - Research Implementation Challenge

This repository contains an independent implementation of the architecture proposed in "Learning Representations by Back-Propagating Errors" (Rumelhart, Hinton & Williams, 1986), specifically focusing on the symmetry detection problem.

## Directory Structure
* `PAPER_NOTES.md`: Theoretical breakdown of the paper's core claims and architecture.
* `results/`: Contains execution logs proving convergence and the ablation study results.
* `src/`: Contains the core PyTorch implementation:
  * `model.py`: Defines the neural network architecture (feedforward network with adjustable hidden nodes).
  * `dataset.py`: Generates the exact symmetry dataset used in the 1986 paper (binary sequences where the left side mirrors the right).
  * `train.py`: The primary training loop for the standard model.
  * `experiment.py`: The script that runs the ablation study across different architectures.

## Dependencies
This project was built using standard deep learning libraries. To run the code, ensure you have the following installed:
* Python 3.x
* PyTorch
* NumPy

## How to Run

**1. Core Implementation (Training)**
To train the standard network on the symmetry dataset, run the following command from the root directory:

    python src/train.py

*What to expect:* The script will output the loss over 20,000 epochs. You will see the network break the symmetry trap and the loss converge to near zero.

**2. Bonus: Ablation Study**
To run the ablation study testing the paper's claim regarding the necessity of hidden nodes, run:

    python src/experiment.py

*What to expect:* The script will test architectures with 1, 2, and 4 hidden nodes, proving that a single node fails to draw the necessary mathematical boundary, while 2+ nodes succeed.