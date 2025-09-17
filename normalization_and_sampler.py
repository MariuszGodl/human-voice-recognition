import numpy as np

def normalize_audio(audio):
    return audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio


def seconds_to_samples(start, end, sr):
    sample_start = int(start * sr)
    sample_end = int(end * sr)
    return sample_start, sample_end