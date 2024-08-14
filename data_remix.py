from sklearn.model_selection import train_test_split
import pandas as pd
import os
import librosa
import numpy as np

col_names = ['Audio', 'Country']
us = pd.DataFrame(columns=col_names)
uk = pd.DataFrame(columns=col_names)
au = pd.DataFrame(columns=col_names)

us_path = "audio/US"
uk_path = "audio/UK"
au_path = "audio/AU"

us_audios = os.listdir(us_path)
uk_audios = os.listdir(uk_path)
au_audios = os.listdir(au_path)

def getAudio(path, audio_list, df, country):
    for file in audio_list:
        audio, sr = librosa.load(os.path.join(path, file), sr=None)
        length = int(librosa.get_duration(y=audio, sr=sr))
        mean = np.mean(audio)
        std_dev = np.std(audio)
        normalized_audio = (audio - mean) / std_dev

        interval_count = 4000 * length
        interval_size = (sr * length) // interval_count

        max_values = []

        for i in range(interval_count):
            start_index = i * interval_size
            end_index = start_index + interval_size
            sub_interval = normalized_audio[start_index:end_index]
            max_values.append(np.max(sub_interval))
        
        df._append({"Audio": max_values, "Country": country}, ignore_index=True)


getAudio(us_path, us_audios, us, 0)
getAudio(uk_path, uk_audios, uk, 1)
getAudio(au_path, au_audios, au, 2)

