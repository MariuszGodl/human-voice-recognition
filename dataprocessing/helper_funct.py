import numpy as np
import string

def normalize_audio(audio):
    return audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio


def seconds_to_samples(start, end, sr):
    sample_start = int(start * sr)
    sample_end = int(end * sr)
    return sample_start, sample_end


def check_if_word_contains_illegal_chars(word):
    # allowed characters: English letters (a–z, A–Z) and maybe apostrophes/hyphens if you want
    allowed_chars = set(string.ascii_letters + "'-")
    
    for char in word:
        if char not in allowed_chars:
            return True  # illegal character found
    return False  # all characters are valid


FORBIDDEN_ENDINGS = [
    '-', '_', '/', '=', '+', '.', ',', ';', ':', '!', '?',
    '(', ')', '[', ']', '{', '}', '<', '>', '"', "'", '|',
    '\\', '*', '&', '^', '%', '$', '#', '@', '~', '`'
]

def strip_endings(word):
    while word and word[-1] in FORBIDDEN_ENDINGS:
        word = word[:-1]
    while word and word[0] in FORBIDDEN_ENDINGS:
        word = word[1:]
    return word