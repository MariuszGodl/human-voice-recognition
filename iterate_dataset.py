import os
import librosa
from textgrid import TextGrid
from const import *


def iterate_dataset(func):
    """
    Decorator to iterate over dataset -> author -> wav+TextGrid files.
    Calls the decorated function with (wav_file, textgrid_file, audio_sample, sr, tg).
    """
    def wrapper(*args, **kwargs):
        for dataset in os.listdir(PATH_RAW_DATASET):
            path_set = os.path.join(PATH_RAW_DATASET, dataset, dataset)

            for author in os.listdir(path_set):
                path_author = os.path.join(path_set, author)

                if os.path.isdir(path_author):
                    for file in os.listdir(path_author):
                        if file.endswith('.wav'):
                            wav_file = os.path.join(path_author, file)

                            textgrid_file = wav_file.replace('.wav', '.TextGrid')
                            if os.path.exists(textgrid_file):
                                tg = TextGrid.fromFile(textgrid_file)
                                audio_sample, sr = librosa.load(wav_file, sr=SAMPLE_RATE)

                                func(author, wav_file, textgrid_file, audio_sample, sr, tg, *args, **kwargs)
    return wrapper
