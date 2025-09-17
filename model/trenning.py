import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from speech_dataset import SpeechDataset
from model import SimpleSpeechModel
import torch
import torch.optim

TEST_SIZE = 0.3
PATH_PROCESSED_DATA = 'data/processed_512/'


labels_df = pd.read_csv(os.path.join( PATH_PROCESSED_DATA, 'labels.csv'), sep='|', names=['tag', 'word','duration', 'author'])

le = LabelEncoder()
labels_df['word_id'] = le.fit_transform(labels_df['word'])

train_df, val_df = train_test_split(labels_df, test_size=0.2, stratify=labels_df["word_id"], random_state=42)


#print me the words that occure less then x times in [word, id,]
# print(labels_df['word_id'].value_counts())

train_dataset = SpeechDataset(train_df, PATH_PROCESSED_DATA, le)
val_dataset = SpeechDataset(val_df, PATH_PROCESSED_DATA, le)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleSpeechModel(n_classes=len(le.classes_)).to(device)
