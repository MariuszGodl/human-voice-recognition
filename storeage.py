import numpy as np
import os

stats = np.load(os.path.join('data/processed/', "mfcc_norm_stats.npz"))
mean = stats["mean"]
std = stats["std"]

stats = np.load("mfcc_norm_copy.npz")
mean1 = stats["mean"]
std1 = stats["std"]

print(mean, std)
print(mean1, std1)