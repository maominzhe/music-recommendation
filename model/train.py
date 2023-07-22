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
from collections import namedtuple, OrderedDict


from utils.dataset import GTZANDataset
from utils.manage import RunBuilder, RunManager
from model import AlexNet

#%%
# global para
torch.set_printoptions(linewidth=120)
ANNOTATIONS_FILE = 'input/GTZAN/features_30_sec.csv'
ANNOTATIONS_FILE_TRAIN = 'input/GTZAN/features_30_sec_train.csv'
ANNOTATIONS_FILE_TEST = 'input/GTZAN/features_30_sec_test.csv'
AUDIO_DIR = 'input/GTZAN/genres_original'

SAMPLE_RATE = 22050
NUM_SAMPLES = 22050 * 5 
plot = False
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
train_df = pd.read_csv(ANNOTATIONS_FILE_TRAIN)
test_df = pd.read_csv(ANNOTATIONS_FILE_TEST)
train_df.info()
test_df.info()
# %%
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print(f'Currently running on device : {device}')

mfcc = torchaudio.transforms.MFCC(
    sample_rate=SAMPLE_RATE,
    n_mfcc=40,
    log_mels=True
)

mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=512,
    n_mels=64
)

gtzan = GTZANDataset(
    ANNOTATIONS_FILE_TRAIN,
    AUDIO_DIR,
    mel_spectrogram,
    SAMPLE_RATE,
    NUM_SAMPLES,
    device
)

print(f'There are {len(gtzan)} samples in the training set')

if torch.cuda.is_available():
    alex = AlexNet().to('cuda')
else:
    alex =  AlexNet().to('cpu')

summary(alex, (1,64,216))
# %%
torch.manual_seed(128)
params = OrderedDict(
    lr = [.001, .0001],
    batch_size = [64],
    num_workers = [0],
    device = ['cuda']
)
# %%
m = RunManager()

for  run in RunBuilder.get_runs(params):
    usd = GTZANDataset(
        ANNOTATIONS_FILE_TRAIN,
        AUDIO_DIR,
        mel_spectrogram,
        SAMPLE_RATE,
        NUM_SAMPLES,
        run.device
    )

    usd_test = GTZANDataset(
        ANNOTATIONS_FILE_TEST,
        AUDIO_DIR,
        mel_spectrogram,
        SAMPLE_RATE,
        NUM_SAMPLES,
        run.device
    )

    print(run)
    device = torch.device(run.device)

    train_data_loader = DataLoader(
        usd,
        batch_size=run.batch_size,
        num_workers=run.num_workers,
        shuffle=True
    )

    test_data_loader = DataLoader(
        usd_test,
        batch_size=run.batch_size,
        num_workers=run.num_workers
    )

    network = alex

    optimizer = optim.Adam(network.parameters(), lr=run.lr)
    m.begin_run(run, network,train_data_loader,test_data_loader)
    # init loss as inf
    best_loss = float('inf')
    for epoch in range(100):
        network.train()
        m.begin_epoch()
        for batch in train_data_loader:
            input = batch[0].to(device)
            target = batch[1].to(device)
            preds = network(input)
            loss = F.cross_entropy(preds,target)
            optimizer.zero_grad()
            #backp propagation
            loss.backward()
            optimizer.step()
            m.track_loss(loss, batch)
            m.track_num_correct(preds, target)

        with torch.no_grad():
            with torch.autocast('cuda'):
                for test_batch in test_data_loader:
                    test_input = batch[0].to(device)
                    test_target = batch[1].to(device)
                    test_preds = network(test_input)
                    test_loss = F.cross_entropy(test_preds, test_target)

                    m.test_loss(test_loss, test_batch)

                    m.test_num_correct(test_preds, test_target)
        
        m.end_epoch()
    
    torch.save(network.state_dict(), f'best_model_okk.pth')
    m.end_run()
    m.save(f'{run.lr}_{run.batch_size}')


# %%
