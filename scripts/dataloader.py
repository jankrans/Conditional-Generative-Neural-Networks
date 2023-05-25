from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
    
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, index):
        x = self.x_data[index]
        y = self.y_data[index]
        return x, y