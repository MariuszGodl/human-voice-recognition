
import torch

import matplotlib.pyplot as plt
Path = '/media/mariusz/Projects/Code/Human_voice_processing/data/processed/words/zaawansowanego/zaawansowanego_0.pt'

features = torch.load(Path)
mfcc = features["mfcc"]
mel_spec = features["mel_spec"]


# Convert to numpy (if tensor)
mel_spec_np = mel_spec.numpy()

# Plot mel spectrogram
plt.figure(figsize=(10, 4))
plt.imshow(mel_spec_np, aspect='auto', origin='lower')
plt.colorbar(label="Amplitude")
plt.title("Mel Spectrogram")
plt.xlabel("Time")
plt.ylabel("Mel Frequency bins")
plt.tight_layout()
plt.show()