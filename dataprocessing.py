from textgrid import TextGrid
from RemovePolichChars import strip_polish_chars
from helper_funct import normalize_audio, seconds_to_samples, check_if_word_contains_illegal_chars
from iterate_dataset import iterate_dataset
import pandas as pd
import numpy as np
import librosa
import string
import torch
import os
from time import sleep
from const import *


@iterate_dataset
def process_data(author, wav_file, textgrid_file, audio_sample, sr, tg):
    
    global stats, mean, std

    size = len(tg[0].intervals)
    labels = [[0 for x in range(4)] for y in range(size)] 

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
                        


if __name__ == '__main__':
    stats = np.load(os.path.join(PATH_PROCESSED_DATA, "mfcc_norm_stats.npz"))
    mean = stats["mean"]
    std = stats["std"]


    process_data()


