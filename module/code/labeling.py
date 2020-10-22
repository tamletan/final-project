import os
import re
import pandas as pd
import numpy as np

import utility as utl

def clean_body(s):
	s = re.sub(r'([\;\:\|•«])', ' ', s) 	# Remove special characters
	s = re.sub(r'(@.*?)[\s]', ' ', s) 		# Remove '@name'
	s = re.sub(r'&amp;', '&', s)  			# Replace '&amp;' with '&'
	s = re.sub('[\r\n]+','\r\n',s).strip()	# Replace multi newline
	s = re.sub('[-]+','-',s).strip()		# Replace multi character '-'
	return s

def batch_mail(df, old):
	labels = []
	for row in df.iterrows():
		# pass row had been labeling
		if old > 0:
			old -= 1
			continue

		os.system('cls')
		body = clean_body(row[1]['body'])
		print('From: {:}\n{:}\n'.format(row[1]['From'], '#'*100))
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

if __name__ == '__main__':
	path = r'..\data\gmail.csv'
	label_log = r'..\log\label.log'

	labels = utl.read_log(label_log)
	df = pd.read_csv(path)
	labels.extend(batch_mail(df, len(labels)))

	os.system('cls')
	print('Total labels: {:}'.format(len(labels)))

	utl.write_log(label_log, labels)
