import string
import nltk
import re
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
from timeit import default_timer as timer

def clean_content(s):
	"""Given a sentence remove its punctuation and stop words"""
	if not isinstance(s,str):
	  s = str(s)																				# Convert to string
	s = s.lower()																				# Convert to lowercase
	s = s.translate(str.maketrans('','',string.punctuation))									# Remove punctuation
	s = re.sub(r'([\;\:\|•«\n])', ' ', s)														# Remove special characters
	s = re.sub(r'(@.*?)[\s]', ' ', s)															# Remove '@name'
	s = re.sub(r'&amp;', '&', s)																# Replace '&amp;' with '&'
	
	tokens = word_tokenize(s)
	stop_words = stopwords.words('english')
	cleaned_s = ' '.join([w for w in tokens if w not in stop_words or w in ['not', 'can']])		# Remove stop-words
	cleaned_s = re.sub(r'\s+', ' ', cleaned_s).strip()											# Replace multi whitespace with single whitespace
	return cleaned_s

def load_data(path):
	df = pd.read_csv(path)
	print('Number of training rows: {:,}\n'.format(df.shape[0]))
	# check class distribution
	print(df['tag'].value_counts(normalize = True))
	
	df["tag"].replace({"legit": 0, "spam":1}, inplace=True)
	print('\nDrop {:,} row with null value\n'.format(df['body'].isnull().sum()))
	df.dropna(subset=['body'], inplace=True)
	print('Number of remain rows: {:,}\n'.format(df.shape[0]))

	start = timer()
	print(format("Clean data", '18s'), end='...')
	df['body'] = df['body'].apply(clean_content)
	print(" Elapsed time: {:.3f}".format(timer()-start))
	return df
