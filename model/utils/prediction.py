import torch
import torch.nn as nn
import random 
import torchaudio

from model import AlexNet


# X输入mel频谱的张量，y实际的标签下标，class_mapping 标签字典
def predict(model, X, y, class_mapping):
    model.eval()    # train <-> eval: changes how model behave (e.g. no dropout, ...)
    with torch.no_grad():
        predictions = model(X)
        # tensor (1, 10) -> [ [0.1, 0.04, ..., 0.6] ]
        # 取出输出最大的下标
        predicted_index = predictions[0].argmax(0)
        # 读出预测标签
        predicted = class_mapping[predicted_index]
        # 实际的标签
        expected = class_mapping[y]
        
    return predicted, expected


# 测试下测试集上的精度
def verify_acc(model_path, sample_rate, gtzan,device):

    class_mapping = [
        'blues',
        'classical',
        'country',
        'disco',
        'hiphop',
        'jazz',
        'metal',
        'pop',
        'reggae',
        'rock'
    ]

    cnn = AlexNet().to(device)
    state_dict = torch.load(model_path)
    cnn.load_state_dict(state_dict)

    # load gtzan validation dataset
    mfcc = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=128,
        log_mels=True
    )

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    count = 0
    for i in range(0,100):
        index = i

        # get a sample from the gtzan dataset for inference
        X, y = gtzan[index][0], gtzan[index][1] # [batch_size, num_channels, freq, time]
        X.unsqueeze_(0) # insert an extra dimension at index 0
        #print(X.shape)
        #print(y)
        print(f'input: {gtzan._get_audio_sample_path(index)}')
        # make an inference
        predicted, expected = predict(cnn, X, y, class_mapping)
        print(f"Predicted: {predicted}")
        print(f"Expected: {expected}")
        if predicted == expected:
            
            count += 1
            #print(count)
    print(count/100.00)
    return (count/100.00) 