from textgrid import TextGrid
from RemovePolichChars import strip_polish_chars
from helper_funct import normalize_audio, seconds_to_samples, check_if_word_contains_illegal_chars, strip_endings
from iterate_dataset import iterate_dataset
from dotenv import load_dotenv
from api_to_chtp import gtp_check_if_it_is_real_word
from augmentation import augment_audio
import pandas as pd
import numpy as np
import librosa
import string
import torch
import os
from time import sleep
from const import *


def process_audio(word_audio, word, labels, duration, author) -> string:
    mfcc = librosa.feature.mfcc(y=word_audio, sr=SAMPLE_RATE, n_mfcc=13, n_fft=NFFT)
    mfcc_norm = (mfcc - mean[:, None]) / std[:, None]
    mfcc_tensor = torch.tensor(mfcc_norm)


    mel_spec = librosa.feature.melspectrogram(
        y=word_audio,
        sr=SAMPLE_RATE,
        n_fft=NFFT,       # larger FFT for better freq resolution
        hop_length=256,   # ~16 ms hop at 16kHz
        n_mels=80,        # higher resolution (standard in speech models)
        fmin=20,          # cut low freqs
        fmax=8000         # limit to speech band (for 16kHz audio)
    )

    # Convert to log scale (better for neural nets)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    # Normalize
    log_mel_spec_norm = (log_mel_spec - np.mean(log_mel_spec)) / np.std(log_mel_spec)
    log_mel_tensor = torch.tensor(log_mel_spec_norm)

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


    features = {
        "mfcc": mfcc_tensor,
        "mel_spec": log_mel_tensor
    }

    torch.save(features, tensor_file_name)
    labels.append([tag, word, duration,author])


@iterate_dataset
def process_data(author, wav_file, textgrid_file, audio_sample, sr, tg):
    
    global stats, mean, std
    size = len(tg[0].intervals)
    labels = [] 

    for i, interval in enumerate(tg[0].intervals):
        duration = interval.maxTime - interval.minTime

        if duration < 0.05 or interval.mark == '':
            continue

        if check_if_word_contains_illegal_chars(interval.mark):  
            continue

        start, end = seconds_to_samples(interval.minTime, interval.maxTime, SAMPLE_RATE)

        word = strip_polish_chars(interval.mark)
        word = strip_endings(word)

        is_real, corrected_word = gtp_check_if_it_is_real_word(word, 'Polish')

        if not is_real:
            if corrected_word != None:
                word = corrected_word
            else:
                continue

        word_audio = audio_sample[start:end]
        if len(word_audio) < NFFT: 
            continue

        word_audio = normalize_audio(word_audio)
        augmented_audios = augment_audio(word_audio)
        
        process_audio(word_audio, word, labels, duration, author)
        for audio in augmented_audios:
            process_audio(audio, word, labels, duration, author)

    with open(PATH_LABEL, 'a') as file:
        for row in labels:
            if row[0] != 0:
                file.write(f"{row[0]}|{row[1]}|{row[2]}|{row[3]}\n")
                        

if __name__ == '__main__':

    load_dotenv()

    stats = np.load(os.path.join(PATH_PROCESSED_DATA, "mfcc_norm_stats.npz"))
    mean = stats["mean"]
    std = stats["std"]


    process_data()


