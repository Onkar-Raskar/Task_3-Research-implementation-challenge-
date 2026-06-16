import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import the classes we just wrote
from dataset import SymmetryDataset
from model import BackpropNetwork

def train_model():
    print("--- Initializing 1986 Backpropagation Training ---")
    
    dataset = SymmetryDataset()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = BackpropNetwork(input_size=6, hidden_size=2, output_size=1)
    
    criterion = nn.MSELoss() 
    optimizer = optim.SGD(model.parameters(), lr=0.5, momentum=0.9)
    
    epochs = 20000 
    
    print("Starting training...")
    
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            
            predictions = model(inputs)
            
            loss = criterion(predictions, labels)
            
            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()
            
        if (epoch + 1) % 200 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] | Error (Loss): {loss.item():.4f}")

    print("\nTraining Complete! The network has learned the internal representation.")
    return model

if __name__ == "__main__":
    trained_model = train_model()