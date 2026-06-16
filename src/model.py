import torch
import torch.nn as nn

class BackpropNetwork(nn.Module):
    """
    A PyTorch implementation of the 1986 feed-forward neural network 
    described by Rumelhart, Hinton, and Williams.
    """
    def __init__(self, input_size=6, hidden_size=2, output_size=1):
        super(BackpropNetwork, self).__init__()
        
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        hidden_sum = self.hidden_layer(x)
        
        hidden_activation = torch.sigmoid(hidden_sum)
        
        output_sum = self.output_layer(hidden_activation)
        
        final_prediction = torch.sigmoid(output_sum)
        
        return final_prediction

if __name__ == "__main__":
    model = BackpropNetwork()
    print(model)
    
    dummy_input = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    
    output = model(dummy_input)
    
    print(f"\nInitial random prediction: {output.item():.4f}")
    print("Notice how it's basically guessing ~0.5 because the weights are random.")