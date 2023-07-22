import torch
import torchaudio
import pandas as pd
import os

from torch.utils.data import Dataset


# 数据预处理类
# 这边的注释
class GTZANDataset(Dataset):
    def __init__(self,
                 annotations_file,
                 audio_dir,
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device):
        # 读取标签文件
        self.annotations = pd.read_csv(annotations_file)
        # 读取音频地址
        self.audio_dir = audio_dir
        # 设置设备
        self.device = device
        # 加梅尔频谱数据加载到设备中
        self.transformation = transformation.to(self.device)
        # 设定采样频率
        self.target_sample_rate = target_sample_rate
        # 设定采样数量
        self.num_samples = num_samples
        
        
    # 返回有多少个音频文件
    def __len__(self):
        return len(self.annotations)

    
    # # 数组的方式可获得音频的数据、标签、路径
    # def __getitem__(self, index):
    #     # 获得歌曲路径
    #     audio_sample_path = self._get_audio_sample_path(index)
    #     # 获得标签
    #     label = self._get_audio_sample_label(index)
    #     #label = torch.type(torch.LongTensor)
    #     #label = torch.
    #     # signal 采样信号 sr 采样频率
    #     signal, sr = torchaudio.load(audio_sample_path)
    #     signal = signal.to(self.device)
    #     # 控制采样频率
    #     signal = self._resample_if_necessary(signal, sr)
    #     # 双通道->单通道
    #     signal = self._mix_down_if_necessary(signal)
    #     # 控制采样数量
    #     signal = self._cut_if_necessary(signal)
    #     signal = self._right_pad_if_necessary(signal)
    #     # 转化下mel频谱
    #     signal = self.transformation(signal)
    #     return signal, label, audio_sample_path

    def __getitem__(self, index):
        # 获得歌曲路径
        audio_sample_path = self._get_audio_sample_path(index)
        # 获得标签
        label = self._get_audio_sample_label(index)
        # signal 采样信号 sr 采样频率
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        # 控制采样频率
        signal = self._resample_if_necessary(signal, sr)
        # 双通道->单通道
        signal = self._mix_down_if_necessary(signal)
        # 控制采样数量
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        # 转化下mel频谱
        signal = self.transformation(signal)
        return signal, label, audio_sample_path
        
    # 是否需要对信号裁剪： 如果采数量 > 设定的数量 -> 裁剪
    def _cut_if_necessary(self, signal):
        # print('_cut_if_necessary')
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal
    
    
    # 是否需要对信号补充： 向右填0补充，如果采数量 < 设定的数量 -> 补充
    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        # print('_right_pad_if_necessary')
        if length_signal < self.num_samples:
            
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            # last_dim_padding.to(self.device)
            
            signal = torch.nn.functional.pad(signal, last_dim_padding)

        return signal

    
    # 重新设定采样频率
    def _resample_if_necessary(self, signal, sr):
        # print('_resample_if_necessary')
        # 如果实际的采样频率没有和设定的一致，那么才重新设定
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate).to(self.device)
            signal = resampler(signal)
            # signal = torchaudio.functional.resample(signal, sr, self.target_sample_rate)
            
        return signal


    # 将音频的双通道改为单通道
    def _mix_down_if_necessary(self, signal):
        # print('_mix_down_if_necessary')
        # 通道数大于1 就 取均值变成单通道
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    # 对音频路径进行拼接提取
    def _get_audio_sample_path(self, index):
        # print('_get_audio_sample_path')
        fold = f"{self.annotations.iloc[index, -3]}"  # str label column
        #print(f"fold: {fold}")
        path = os.path.join(self.audio_dir, fold, str(self.annotations.iloc[
            index, 1]))
        #print(path)
        return path
    
    
    # 从csv文件中提取出标签
    def _get_audio_sample_label(self, index):
        # print('_get_audio_sample_label')
        return self.annotations.iloc[index, -2]
    