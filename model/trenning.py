import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from speech_dataset import SpeechDataset
from model import SimpleSpeechModel
import torch.nn as nn
import torch
import torch.optim as optim

TEST_SIZE = 0.3
PATH_PROCESSED_DATA = 'data/processed/'

from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    xs, ys = zip(*batch)
    # pad mel_spec
    mel_specs = [x["mel_spec"].T for x in xs]  # transpose so time is dim 0
    mel_specs_padded = pad_sequence(mel_specs, batch_first=True).transpose(1,2)

    mfccs = [x["mfcc"].T for x in xs]
    mfccs_padded = pad_sequence(mfccs, batch_first=True).transpose(1,2)

    durations = torch.tensor([x["duration"] for x in xs])

    new_x = {
        "mel_spec": mel_specs_padded,
        "mfcc": mfccs_padded,
        "duration": durations
    }
    y = torch.stack(ys)
    return new_x, y


labels_df = pd.read_csv(os.path.join( PATH_PROCESSED_DATA, 'labels.csv'), sep='|', names=['tag', 'word','duration', 'author'])

le = LabelEncoder()
labels_df['word_id'] = le.fit_transform(labels_df['word'])

train_df, val_df = train_test_split(labels_df, test_size=0.2, stratify=labels_df["word_id"], random_state=42)

# train_dataset = SpeechDataset(train_df, PATH_PROCESSED_DATA, le)
# val_dataset = SpeechDataset(val_df, PATH_PROCESSED_DATA, le)


# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
# val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)

full_dataset = SpeechDataset(labels_df, PATH_PROCESSED_DATA, le)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleSpeechModel(n_classes=len(le.classes_)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(100):

    train_df, val_df = train_test_split(
        labels_df, 
        test_size=0.2, 
        stratify=labels_df["word_id"], 
        random_state=None  # None ensures a new split each time
    )

    train_dataset = SpeechDataset(train_df, PATH_PROCESSED_DATA, le)
    val_dataset = SpeechDataset(val_df, PATH_PROCESSED_DATA, le)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)
    # --- Training ---
    model.train()
    train_correct, train_total, train_loss = 0, 0, 0
    for batch_idx, batch in enumerate(train_loader, start=1):
        x, y = batch
        x = {k: v.to(device) for k, v in x.items()}
        y = y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        # accumulate stats
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == y).sum().item()
        train_total += y.size(0)

        # print batch stats
        batch_acc = 100 * train_correct / train_total
        avg_batch_loss = train_loss / batch_idx
        print(f"Epoch {epoch+1}, Batch {batch_idx}: "
              f"Train Loss={avg_batch_loss:.4f}, Train Acc={batch_acc:.2f}%")

    # --- Validation ---
    model.eval()
    val_correct, val_total, val_loss = 0, 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x = {k: v.to(device) for k, v in x.items()}
            y = y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == y).sum().item()
            val_total += y.size(0)

    val_acc = 100 * val_correct / val_total
    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1} Summary: "
          f"Train Loss={avg_batch_loss:.4f}, Train Acc={batch_acc:.2f}% | "
          f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.2f}%")

    # After training 
    MODEL_PATH = "models/simple_speech_model.pth" + str(epoch)
    torch.save(model.state_dict(), MODEL_PATH) 
    print(f"Model saved to {MODEL_PATH}") 