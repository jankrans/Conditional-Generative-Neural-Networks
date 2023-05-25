import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NormalizingFlow(nn.Module):
    def __init__(self,base_dist,transforms):
        super(NormalizingFlow, self).__init__()
        self.base_dist = base_dist
        self.transforms = transforms
    
    def forward(self,x):
        z = self.base_dist.sample(x.shape[0])
        log_det = torch.zeros(x.shape[0])
        for transform in self.transforms:
            z, ld = transform(z)
            log_det += ld
        log_prob = self.base_dist.log_prob(z) + log_det
        return z, log_prob
    
    def inverse(self,z):
        for transform in reversed(self.transforms):
            z = transform.inverse(z)
        return z
    
class CustomDataset(Dataset):
    def __init__(self, data, attributes):
        self.data = data
        self.attributes = attributes    

    def __getitem__(self, index):
        return self.data[index], self.attributes[index]
    
    def __len__(self):
        return len(self.data)

    
def train(model, optimizer, data_loader, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x in data_loader:
            x = x.to(device)
            optimizer.zero_grad()
            z, log_prob = model(x)
            loss = -torch.mean