import pandas as pd

#read data into dataframes and reset index because it was messed up
test = pd.read_feather('test.feather')
test = test.reset_index(drop=True)
train = pd.read_feather('train.feather')
train = train.reset_index(drop=True)
val = pd.read_feather('val.feather')
val = val.reset_index(drop=True)

