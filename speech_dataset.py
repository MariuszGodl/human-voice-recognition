from torch.utils.data import Dataset
import os

class SpeechDataset(Dataset):

    def __init__(self, df, data_path, label_encoder):
        self.df = df
        self.data_path = data_path
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        tensor_file = os.path.join(self.data_path, 'words', row['word'], row['tag'] )