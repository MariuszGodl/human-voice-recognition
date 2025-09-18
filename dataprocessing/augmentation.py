import numpy as np
import librosa
from const import *


def augment_with_noise(audio, noise_factor=0.05):
    """Add Gaussian noise to audio."""
    noise = np.random.randn((len(audio)))
    augmented = audio + noise * noise_factor
    return np.clip(augmented)

def pitch_change(audio, n_steps=2):
    """Shift pitch up/down by n_steps semitones."""
    return librosa.effects.pitch_shift(y=audio, sr=SAMPLE_RATE, n_steps=n_steps)

def speed_perturb(audio, rate=0.9):
    """Speed up or slow down the audio."""
    return librosa.effects.time_stretch(y=audio, rate=rate)

def augment_audio(audio):
    n_of_copies = np.random.randint(1, 3)
    augmented_audios = []

    augmentation_features = {
        "noise" : lambda x: augment_with_noise(x, np.random.uniform(0.01, 0.03)),
        "pitch" : lambda x: pitch_change(x, np.random.randint(-2,3)),
        "speed" : lambda x:speed_perturb(x, np.random.uniform(0.9, 1.1))
    }
    
    for _ in range(n_of_copies):
        audio_copy = audio.copy()

        fetures_to_apply = np.random.choice(
            list(augmentation_features.keys()),
            size=np.random.randint(1, len(augmentation_features) + 1),
            replace=False
        )

        for feature in fetures_to_apply:
            audio_copy = augmentation_features[feature](audio_copy)

        augmented_audios.append(audio_copy)
    
    return augmented_audios
