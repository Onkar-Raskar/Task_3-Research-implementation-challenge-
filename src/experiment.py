import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import SymmetryDataset
from model import BackpropNetwork

torch.manual_seed(42)

def run_experiment(hidden_nodes, epochs=20000): 
    dataset = SymmetryDataset()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = BackpropNetwork(input_size=6, hidden_size=hidden_nodes, output_size=1)
    
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.5, momentum=0.9)
    
    final_loss = 0.0
    
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            predictions = model(inputs)
            loss = criterion(predictions, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            final_loss = loss.item()
            
    return final_loss

if __name__ == "__main__":
    print("--- 1986 Paper Verification: Architecture Sweep ---")
    print("Testing the impact of hidden layer size on symmetry detection.\n")
    
    nodes_to_test = [1, 2, 4]
    
    print(f"{'Hidden Nodes':<15} | {'Final Loss (after 20000 epochs)':<30}")
    print("-" * 50)
    
    for nodes in nodes_to_test:
        loss = run_experiment(nodes)
        print(f"{nodes:<15} | {loss:.4f}")