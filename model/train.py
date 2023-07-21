# import libraries

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
from itertools import product
from collections import namedtuple
from collections import OrderedDict
from IPython.display import display,clear_output
import time
import json
from torchsummary import summary
import matplotlib.pyplot as plt



# global para
torch.set_printoptions(linewidth=120)
ANNOTATIONS_FILE = '/input/GTZAN/features_30_sec.csv'

# data preprocessing
