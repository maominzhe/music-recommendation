import torch
import torchaudio
import pandas as pd
import os

from torch.utils.data import Dataset

class GTZANDataset(Dataset):
    def __init__(self,
                 annotations_file,
                 audio_dir,
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device):
        # Read label file
        self.annotations = pd.read_csv(annotations_file)
        # Read audio dir
        self.audio_dir = audio_dir
        # Read device
        self.device = device
        # spectrum data loaded into the device
        self.transformation = transformation.to(self.device)
        # hyper params
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        
        
    def __len__(self):
        return len(self.annotations)

    
    def __getitem__(self, index):
        # get audio path
        audio_sample_path = self._get_audio_sample_path(index)
        # get label
        label = self._get_audio_sample_label(index)
        # signal and sample rate sr 
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        # Control Sampling Frequency
        signal = self._resample_if_necessary(signal, sr)
        # Dual Channel -> Single Channel
        signal = self._mix_down_if_necessary(signal)
        # Controlling the number of samples
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        # mel transformation
        signal = self.transformation(signal)
        return signal, label, audio_sample_path
        
    # Whether the signal needs to be cropped: if the number of picks > the set number -> crop
    def _cut_if_necessary(self, signal):
        # print('_cut_if_necessary')
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal
    
    
    # Whether the signal needs to be replenished: 0 replenishment to the right, if the number of picks < set number -> replenishment
    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        # print('_right_pad_if_necessary')
        if length_signal < self.num_samples:
            
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            # last_dim_padding.to(self.device)
            
            signal = torch.nn.functional.pad(signal, last_dim_padding)

        return signal

    
    # Reset sampling frequency
    def _resample_if_necessary(self, signal, sr):
        # print('_resample_if_necessary')
        # If the actual sampling frequency does not match the setting, then only reset it.
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate).to(self.device)
            signal = resampler(signal)
            # signal = torchaudio.functional.resample(signal, sr, self.target_sample_rate)
            
        return signal


    # Change dual channels of audio to single channel
    def _mix_down_if_necessary(self, signal):
        # print('_mix_down_if_necessary')
        # If the number of channels is greater than one, then take the average and make it a single channel.
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    # Splicing and extracting audio paths
    def _get_audio_sample_path(self, index):
        # print('_get_audio_sample_path')
        fold = f"{self.annotations.iloc[index, -3]}"  # str label column
        #print(f"fold: {fold}")
        path = os.path.join(self.audio_dir, fold, str(self.annotations.iloc[
            index, 1]))
        #print(path)
        return path
    
    
    # read label from csv
    def _get_audio_sample_label(self, index):
        # print('_get_audio_sample_label')
        return self.annotations.iloc[index, -2]
    