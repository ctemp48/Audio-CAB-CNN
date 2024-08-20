import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

#read data into dataframes and reset index because it was messed up
test = pd.read_feather('test.feather')
test = test.reset_index(drop=True)
train = pd.read_feather('train.feather')
train = train.reset_index(drop=True)
val = pd.read_feather('val.feather')
val = val.reset_index(drop=True)


#create the dataset class
class audioDataset(Dataset):
    def __init__(self, df):
        #we must conver the ndarray to type of float32
        self.x = torch.from_numpy(np.vstack(df['Audio'].to_numpy()).astype(np.float32))
        self.y = torch.from_numpy(df['Country'].to_numpy())

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

test_set = audioDataset(test)
train_set = audioDataset(train)
val_set = audioDataset(val)

test_dataloader = DataLoader(test_set, batch_size=128, shuffle=True)
train_dataloader = DataLoader(train_set, batch_size=128, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=128, shuffle=True)