from textgrid import TextGrid
from RemovePolichChars import strip_polish_chars
from normalization_and_sampler import normalize_audio, seconds_to_samples
import pandas as pd
import numpy as np
import librosa
import string
import torch
import os
from time import sleep

PATH_RAW_DATASET = 'data/raw/'
PATH_PROCESSED_DATA = 'data/processed'
PATH_LABEL = 'data/processed/labels.csv'
SAMPLE_RATE = 16000 # Hz
NFFT = 512
HOP_LENGTH = 256

# add spectrographs


def check_if_word_contains_illegal_chars(word):
    # allowed characters: English letters (a–z, A–Z) and maybe apostrophes/hyphens if you want
    allowed_chars = set(string.ascii_letters + "'-")
    
    for char in word:
        if char not in allowed_chars:
            return True  # illegal character found
    return False  # all characters are valid

stats = np.load(os.path.join(PATH_PROCESSED_DATA, "mfcc_norm_stats.npz"))
mean = stats["mean"]
std = stats["std"]


for dataset in os.listdir(PATH_RAW_DATASET):
    path_set = os.path.join(PATH_RAW_DATASET, dataset, dataset)

    for author in os.listdir(path_set):

        path_author = os.path.join(path_set, author)

        if os.path.isdir(path_author):

            for file in os.listdir(path_author):

                if file.endswith('.wav'):
                    wav_file = os.path.join(path_author, file)

                    textgrid_file = file.replace('.wav', '.TextGrid')
                    textgrid_file = os.path.join(path_author, textgrid_file)
                    if os.path.exists(textgrid_file):
                        tg = TextGrid.fromFile(textgrid_file)
                        size = len(tg[0].intervals)

                        labels = [[0 for x in range(4)] for y in range(size)] 
                        audio_sample, sr = librosa.load(wav_file, sr=SAMPLE_RATE)

                        for i, interval in enumerate(tg[0].intervals):
                            duration = interval.maxTime - interval.minTime
                            if duration < 0.05 or interval.mark == '':
                                continue
       
                            if check_if_word_contains_illegal_chars(interval.mark):  
                                continue
                            start, end = seconds_to_samples(interval.minTime, interval.maxTime, SAMPLE_RATE)
                            # strip endings of the words from -,
                            # think about adding chatgtp api to tell if it is accualy a polish word 
                            # but rather for folder creation not for each entry, and store it for further usage names 
                            # not appproved by api to classify them by hand and if necessery add redirection to diffrent folder
                            # count number of used tokens to asses the $$
                            word = strip_polish_chars(interval.mark)

                            #add spectrographs for better model quality
                            print(textgrid_file, word)
                            word_audio = audio_sample[start:end]
                            if len(word_audio) < NFFT: 
                                continue

                            word_audio = normalize_audio(word_audio)
                            
                            mfcc = librosa.feature.mfcc(y=word_audio, sr=SAMPLE_RATE, n_mfcc=13, n_fft=NFFT)
                            mfcc_norm = (mfcc - mean[:, None]) / std[:, None]
                            mfcc_tensor = torch.tensor(mfcc_norm)

                            word_folder_path = os.path.join(PATH_PROCESSED_DATA, 'words', word)
                            nr_of_occurances = 0
                            tag = word + '_' + str(nr_of_occurances) + '.pt'

                            if os.path.exists(word_folder_path):

                                nr_of_occurances = len(os.listdir(word_folder_path))
                                tag = word + '_' + str(nr_of_occurances) + '.pt'
                                tensor_file_name = os.path.join(word_folder_path, tag)

                            else:
                                os.makedirs(word_folder_path)
                                tensor_file_name = os.path.join(word_folder_path, tag)

                            torch.save(mfcc_tensor, tensor_file_name)

                            labels[i][0]=tag
                            labels[i][1]=word
                            labels[i][2]=duration
                            labels[i][3]=author

                        with open(PATH_LABEL, 'a') as file:
                            for row in labels:
                                if row[0] != 0:
                                    file.write(f"{row[0]}|{row[1]}|{row[2]}|{row[3]}\n")
                        


# Human_voice_processing on  main [!] via  v3.12.3 (venv) 
# ❯ find data/processed/words/ -type f | wc -l
# 88305

# Human_voice_processing on  main [!] via  v3.12.3 (venv) 
# ❯ find data/processed/words/ -type d | wc -l
# 7996

# ❯ tree -L 2
# .
# ├── data
# │   ├── processed
# │   └── raw
# ├── dataprocessing.py
# ├── get_global_parameters.py
# ├── __pycache__
# │   └── RemovePolichChars.cpython-312.pyc
# ├── RemovePolichChars.py
# └── venv
#     ├── bin
#     ├── include
#     ├── lib
#     ├── lib64 -> lib
#     ├── pyvenv.cfg
#     └── share
