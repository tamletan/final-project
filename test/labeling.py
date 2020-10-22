import os
import re
import pandas as pd
import numpy as np

def clean_body(s):
	s = re.sub(r'([\;\:\|•«])', ' ', s) # remove special characters
	s = re.sub(r'(@.*?)[\s]', ' ', s) # Remove '@name'
	s = re.sub(r'&amp;', '&', s)  # Replace '&amp;' with '&'
	s = re.sub('[\r\n]+','\r\n',s).strip()
	s = re.sub('[-]+','-',s).strip()
	return s

def batch_mail(df, old):
	labels = []
	for row in df.iterrows():
		if old > 0:
			old -= 1
			continue

		os.system('cls')
		body = clean_body(row[1]['Body'])
		print('From: {:}\n'.format(row[1]['From']))
		print(body)

		label = taging(row[0])
		if label == 'exit':
			break
		labels.append(label)

	return labels

def taging(index):
	while(True):
		try:
			text = input('\nTag ({:}): '.format(index))
			if text in ['0', '1', 'exit']:
				return text
			else:
				raise
		except:
			print('Only 0 or 1!')

def read_label(path):
	if os.path.isfile(label_csv):
		try:
			df = pd.read_csv(label_csv, dtype=str)
			return [i for i in df['label']]
		except Exception as e:
			return []
	else:
		return []

def save_label(path, data):
	with open(path, 'w') as f:
		value = '\n'.join(data)
		f.write('label\n'+value)

if __name__ == '__main__':
	path = r'.\data\gmail_EN_preds.csv'
	label_csv = r'.\label.csv'

	labels = read_label(label_csv)
	df = pd.read_csv(path)
	labels.extend(batch_mail(df, len(labels)))

	os.system('cls')
	print('Total labels: {:}'.format(len(labels)))
	save_label(label_csv, labels)
