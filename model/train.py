# import libraries
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
import torchaudio
from torch.utils.data import DataLoader
import pandas as pd
import os
from IPython.display import display,clear_output
import time
import json
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np

#%%
# global para
torch.set_printoptions(linewidth=120)
ANNOTATIONS_FILE = 'input/GTZAN/features_30_sec.csv'
TRAIN_FILE = 'input/GTZAN/features_30_sec_train.csv'
TEST_FILE = 'input/GTZAN/features_30_sec_test.csv'
#%%
# Data preprocessing
dataframe = pd.read_csv(ANNOTATIONS_FILE)
dataframe.info()
# %%
# numerize label 
labels = set()
for row in range(len(dataframe)):
    labels.add(dataframe.iloc[row, -1])
labels_list = []
for label in labels:
    labels_list.append(label)
sorted_labels = sorted(labels_list)
sorted_labels
mapping = {}
for index, label in enumerate(sorted_labels):
    mapping[label] = index
dataframe["num_label"] = dataframe["label"]
new_dataframe = dataframe.replace({"num_label": mapping})
error_row = new_dataframe[new_dataframe['filename'] == 'jazz.00054.wav'].index
new_dataframe.drop(error_row, inplace=True)
# %%
# split dataset into training and testing [7:3]
df = new_dataframe
df['split'] = np.random.randn(df.shape[0], 1)

msk = np.random.rand(len(df)) <= 0.7

train = df[msk]
test = df[~msk]
# %%
# save training and testing annotation files
train.to_csv("input/GTZAN/features_30_sec_train.csv")
test.to_csv("input/GTZAN/features_30_sec_test.csv")
# %%
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)
train_df.info()
test_df.info()
# %%

