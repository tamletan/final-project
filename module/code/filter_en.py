from langdetect import detect, detect_langs, DetectorFactory
import os
import re
import pandas as pd
import numpy as np

DetectorFactory.seed = 0

def add_lang(path):
	df = pd.read_csv(path)
	langs = []
	for text in df['body']:
		try:
			lang = detect(text)
			langs.append(lang)
		except Exception as e:
			langs.append(np.nan)

	df['Lang'] = langs
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
	data_path = r'.\data\gmail.csv'
	temp = r'.\data\lang_detect.csv'
	en_only = r'.\data\lang_en.csv'

	df = add_lang(data_path)
	
	en = filter_en(df)
	df.drop(en.index, inplace=True)
	df.to_csv(en_only, index=False)
	print(df['Lang'].value_counts())
