import torch
from torch.utils.data import Dataset, DataLoader
import itertools

class SymmetryDataset(Dataset):
    """
    Generates all 64 possible 6-bit binary arrays and labels them
    1 if they are perfectly symmetrical, and 0 if they are not.
    """
    def __init__(self, sequence_length=6):
        self.sequence_length = sequence_length
        
        all_combinations = list(itertools.product([0.0, 1.0], repeat=sequence_length))
        
        self.data = torch.tensor(all_combinations, dtype=torch.float32)
        
        labels = []
        for seq in all_combinations:
            is_symmetrical = (list(seq) == list(seq)[::-1])
            labels.append([1.0 if is_symmetrical else 0.0])
            
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

if __name__ == "__main__":
    dataset = SymmetryDataset()
    print(f"Total dataset size: {len(dataset)}")
    
    for i in range(5):
        inputs, label = dataset[i]
        print(f"Input: {inputs.numpy()} | Symmetrical: {label.item()}")
    
    idx = 30
    inputs, label = dataset[idx]
    print(f"Input: {inputs.numpy()} | Symmetrical: {label.item()}")