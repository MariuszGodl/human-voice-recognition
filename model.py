import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

TEST_SIZE = 0.3

labels_df = pd.read_csv('data/processed/labels.csv', sep='|', names=['tag', 'word','duration', 'author'])

labels_df = labels_df[~labels_df.apply(lambda row: row.astype(str).str.contains(r'\|ss\|')).any(axis=1)]

le = LabelEncoder()
labels_df['word_id'] = le.fit_transform(labels_df['word'])

train_df, val_df = train_test_split(labels_df, test_size=0.2, stratify=labels_df["word_id"], random_state=42)


#print me the words that occure less then x times in [word, id,]
print(labels_df['word_id'].value_counts())
