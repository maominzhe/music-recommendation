import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio

from utils.dataset import  GTZANDataset
from utils.manage import RunBuilder, RunManager
from utils.prediction import verify_acc, predict
from model import AlexNet

# global para
torch.set_printoptions(linewidth=120)
ANNOTATIONS_FILE = 'input/GTZAN/features_30_sec.csv'
ANNOTATIONS_FILE_TRAIN = 'input/GTZAN/features_30_sec_train.csv'
ANNOTATIONS_FILE_TEST = 'input/GTZAN/features_30_sec_test.csv'
AUDIO_DIR = 'input/GTZAN/genres_original'

SAMPLE_RATE = 22050
NUM_SAMPLES = 22050 * 5 
plot = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# transformation method
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

# load dataset
gtzan = GTZANDataset(
    ANNOTATIONS_FILE_TRAIN,
    AUDIO_DIR,
    mel_spectrogram,
    SAMPLE_RATE,
    NUM_SAMPLES,
    device
)

print(gtzan[0][0])

model_path = "best_model_okk.pth"

verify_acc(model_path,SAMPLE_RATE,gtzan,device)