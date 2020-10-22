from langdetect import detect, detect_langs, DetectorFactory
import os
import re
import pandas as pd
import numpy as np

DetectorFactory.seed = 0

def add_lang(df):
	langs = []
	for text in df['body']:
		try:
			lang = detect(text)
			langs.append(lang)
		except Exception as e:
			langs.append(np.nan)

	df['Lang'] = langs
	print(df['Lang'].value_counts())
	df.dropna(subset=['body','Lang'], inplace=True)
	return df

def filter_lang(df):
	not_VEJ = ~df.Lang.isin(['vi','en','ja'])
	return df[not_VEJ]

def filter_en(df):
	return df[df.Lang != 'en']

def check_rate(text):
	print(detect_langs(text))

if __name__ == '__main__':
	data_path = r'..\data\gmail.csv'
	en_only = r'..\data\gmail_en.csv'

	df = pd.read_csv(data_path)
	df = add_lang(df)
	
	en = filter_en(df)
	df.drop(en.index, inplace=True)
	df.to_csv(en_only, index=False)
	print(df['Lang'].value_counts())
