import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split


def to_string(s):
    if not isinstance(s, str):
        s = str(s)
    return s


def read_data(path):
    df = pd.read_csv(path)
    df['body'] = df['body'].apply(to_string)
    df["tag"].replace({"legit": 0, "spam": 1}, inplace=True)
    return df


def split_data(df, test_path, train_path):
    train_text, test_text, train_labels, test_labels = train_test_split(df['body'], df['tag'], random_state=2020, test_size=0.2, stratify=df['tag'])

    frame1 = {'tag': test_labels, 'body': test_text}
    test_frame = pd.DataFrame(frame1)

    frame2 = {'tag': train_labels, 'body': train_text}
    train_frame = pd.DataFrame(frame2)

    test_frame.to_csv(test_path, index=False)
    train_frame.to_csv(train_path, index=False)


if __name__ == '__main__':
    file = r'.\data\merge.csv'
    test_path = r".\data\test_data.csv"
    train_path = r".\data\train_data.csv"
    df = read_data(file)
    split_data(df, test_path, train_path)
