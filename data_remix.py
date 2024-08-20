import pandas as pd
import os
import librosa
import numpy as np

#create empty dataframes
col_names = ['Audio', 'Country']
us = pd.DataFrame(columns=col_names)
uk = pd.DataFrame(columns=col_names)
au = pd.DataFrame(columns=col_names)

#audio folder paths
us_path = "audio/US"
uk_path = "audio/UK"
au_path = "audio/AU"

#get list of each audio file 
us_audios = os.listdir(us_path)
uk_audios = os.listdir(uk_path)
au_audios = os.listdir(au_path)

def getAudio(path, audio_list, df, country):
    """
    load in each audio file, truncate to 10 seconds, normalize it and then divide into
    4000 sub intervals for each second of audio. take the max from
    each subinterval to use as the input data.
    """
    for file in audio_list:
        audio, sr = librosa.load(os.path.join(path, file), sr=None, duration=10.0)
        length = librosa.get_duration(y=audio, sr=sr)
        mean = np.mean(audio)
        std_dev = np.std(audio)
        normalized_audio = (audio - mean) / std_dev

        interval_count = int(4000 * length)
        interval_size = (sr * length) // interval_count

        max_values = []

        for i in range(interval_count):
            start_index = int(i * interval_size)
            end_index = int(start_index + interval_size)
            sub_interval = normalized_audio[start_index:end_index]
            max_values.append(np.max(sub_interval))
        
        df.loc[len(df.index)] = [max_values, country]

        
getAudio(us_path, us_audios, us, 0)
getAudio(uk_path, uk_audios, uk, 1)
getAudio(au_path, au_audios, au, 2)

#shuffle each dataframe 
us = us.sample(frac=1).reset_index(drop=True)
uk = uk.sample(frac=1).reset_index(drop=True)
au = au.sample(frac=1).reset_index(drop=True)

#seperate into 60% train 10% validation and 30% test
us_train = us.iloc[0:387]
us_val = us.iloc[387:452]
us_test = us.iloc[452:646]

uk_train = uk.iloc[0:200]
uk_val = uk.iloc[200:233]
uk_test = uk.iloc[233:333]

au_train = au.iloc[0:458]
au_val = au.iloc[458:534]
au_test = au.iloc[534:764]

#combine train, validation and test into respective dataframes
train = pd.concat([us_train, uk_train, au_train])
val = pd.concat([us_val, uk_val, au_val])
test = pd.concat([us_test, uk_test, au_test])

#write to feather files
test.to_feather('test.feather')
val.to_feather('val.feather')
train.to_feather('train.feather')