from textgrid import TextGrid
from RemovePolichChars import strip_polish_chars
import pandas as pd
import numpy as np
import librosa
import torch
import os
from time import sleep

PATH_RAW_DATASET = 'data/raw/'
PATH_PROCESSED_DATA = 'data/processed'
PATH_LABEL = 'data/processed/labels.csv'
SAMPLE_RATE = 16000 # Hz

def normalize_audio(audio):
    return audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio


def seconds_to_samples(start, end, sr):
    sample_start = int(start * sr)
    sample_end = int(end * sr)
    return sample_start, sample_end

stats = np.load(os.path.join(PATH_PROCESSED_DATA, "mfcc_norm_stats.npz"))
mean = stats["mean"]
std = stats["std"]


for dataset in os.listdir(PATH_RAW_DATASET):
    dataset = '1-500'
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

                        labels = [[0 for x in range(3)] for y in range(size)] 
                        audio_sample, sr = librosa.load(wav_file, sr=SAMPLE_RATE)

                        for i, interval in enumerate(tg[0].intervals):

                            if interval.maxTime - interval.minTime < 0.05 or interval.mark == '':
                                continue
       

                            start, end = seconds_to_samples(interval.minTime, interval.maxTime, SAMPLE_RATE)
                            
                            word = strip_polish_chars(interval.mark)

                            word_audio = audio_sample[start:end]
                            if len(word_audio) < 2048: 
                                continue

                            word_audio = normalize_audio(word_audio)
                            
                            mfcc = librosa.feature.mfcc(y=word_audio, sr=SAMPLE_RATE, n_mfcc=13)
                            mfcc_tensor = torch.tensor(mfcc)

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
                            labels[i][2]=author
                            labels[i][3]=start / SAMPLE_RATE
                            labels[i][4]=end / SAMPLE_RATE

                        with open(PATH_LABEL, 'a') as file:
                            for row in labels:
                                if row[0] != 0:
                                    file.write(f"{row[0]}|{row[1]}|{row[2]}|{row[3]}|{row[4]}\n")
                        
                        #sleep(10)


#first create word recognition


#second word splitter

