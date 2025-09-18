from iterate_dataset import iterate_dataset
from helper_funct import normalize_audio, seconds_to_samples
from RemovePolichChars import strip_polish_chars
from textgrid import TextGrid
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import librosa
import torch
import os
from time import sleep
from const import *



@iterate_dataset
def get_global_parameters(author, wav_file, textgrid_file, audio_sample, sr, tg):

    global global_sum, global_sq_sum, total_frames, k  

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

        if global_sum is None:
            global_sum = np.sum(mfcc, axis=1)
            global_sq_sum = np.sum(mfcc ** 2, axis=1)
        else:
            global_sum += np.sum(mfcc, axis=1)
            global_sq_sum += np.sum(mfcc ** 2, axis=1)

        total_frames += mfcc.shape[1]
        k+=1
        print(k)



if __name__ == '__main__':
    k = 0   
    global_sum = None
    global_sq_sum = None
    total_frames = 0

    load_dotenv()

    get_global_parameters()             

    global_mean = global_sum / total_frames
    global_var = (global_sq_sum / total_frames) - (global_mean ** 2)
    global_std = np.sqrt(global_var)

    np.savez(os.path.join(PATH_PROCESSED_PARAMETERS, "mfcc_norm_stats.npz"),
            sum=global_sum, sq_sum=global_sq_sum,
            mean=global_mean, std=global_std)


