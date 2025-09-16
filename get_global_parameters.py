
from textgrid import TextGrid
from RemovePolichChars import strip_polish_chars
import pandas as pd
import numpy as np
import librosa
import torch
import os
from time import sleep

PATH_RAW_DATASET = 'data/raw/'
PATH_PROCESSED_PARAMETERS = 'data/processed/'
PATH_LABEL = 'data/processed/labels.csv'
SAMPLE_RATE = 16000 # Hz
NFFT = 512

def normalize_audio(audio):
    return audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio


def seconds_to_samples(start, end, sr):
    sample_start = int(start * sr)
    sample_end = int(end * sr)
    return sample_start, sample_end

global_sum = None
global_sq_sum = None
total_frames = 0

k=0
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

                        labels = [[0 for x in range(3)] for y in range(size)] 
                        audio_sample, sr = librosa.load(wav_file, sr=SAMPLE_RATE)

                        for i, interval in enumerate(tg[0].intervals):

                            if interval.maxTime - interval.minTime < 0.05 or interval.mark == '':
                                continue
       
                            start, end = seconds_to_samples(interval.minTime, interval.maxTime, SAMPLE_RATE)
                            
                            word = strip_polish_chars(interval.mark)

                            word_audio = audio_sample[start:end]

                            if len(word_audio) < NFFT: 
                                continue
                            word_audio = normalize_audio(word_audio)
                            
                            mfcc = librosa.feature.mfcc(y=word_audio, sr=SAMPLE_RATE, n_mfcc=13, n_fft=NFFT)

                            # update global sums
                            if global_sum is None:
                                global_sum = np.sum(mfcc, axis=1)
                                global_sq_sum = np.sum(mfcc ** 2, axis=1)
                            else:
                                global_sum += np.sum(mfcc, axis=1)
                                global_sq_sum += np.sum(mfcc ** 2, axis=1)

                            total_frames += mfcc.shape[1]
                            k+=1
                            print(k)

global_mean = global_sum / total_frames
global_var = (global_sq_sum / total_frames) - (global_mean ** 2)
global_std = np.sqrt(global_var)

                
np.savez(os.path.join(PATH_PROCESSED_PARAMETERS, "mfcc_norm_stats.npz"),
         sum=global_sum, sq_sum=global_sq_sum,
         mean=global_mean, std=global_std)


