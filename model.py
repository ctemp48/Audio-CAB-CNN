import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        #we must convert the ndarray of type object to type of float32
        self.x = torch.from_numpy(np.vstack(df['Audio'].to_numpy()).astype(np.float32))
        self.x.to(device)
        self.y = torch.from_numpy(df['Country'].to_numpy())
        self.y.to(device)

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

class audioClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=4)
        self.activation1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=2)
        self.dropout1 = nn.Dropout(p=0.15)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=4)
        self.activation2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=2)
        self.dropout2 = nn.Dropout(p=0.15)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=10)
        self.activation3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=10, stride=5)
        self.dropout3 = nn.Dropout(p=0.1)
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=128, kernel_size=10)
        self.activation4 = nn.ReLU()
        self.pool4 = nn.MaxPool1d(kernel_size=10, stride=5)
        self.batch_norm = nn.BatchNorm1d(num_features=396)
    
    def forward(self, x):
        x = self.activation1(self.conv1(x))
        x = self.dropout1(self.pool1(x))
        x = self.activation2(self.conv2(x))
        x = self.dropout2(self.pool2(x))
        x = self.activation3(self.conv3(x))
        x = self.dropout3(self.pool3(x))
        x = self.activation4(self.conv4(x))
        x = self.pool4(x)
        x = self.batch_norm(x)
        return x

#when we start training for each x in train loader we must reshape to size of [1, 40000]

model = audioClassifier()
pred = model(test_set[0][0].reshape(1, 40000))
print(pred.shape)
        

        