import os
import re
import pandas as pd

root = r'.\data\merge'
csv_file = r'.\data\merge.csv'

def read_file(path):
	df = pd.read_csv(path)
	return df

def get_file(root):
	f = []
	for dirs, _, files in os.walk(root):
		if files:
			for file in files:
				f.append(os.path.join(dirs,file))
	return f

def load_data(files):
	df = pd.DataFrame(columns=['tag', 'body'])
	for file in files:
		df = df.append(read_file(file), ignore_index=True)
	return df

if __name__ == '__main__':
	files = get_file(root)
	df = load_data(files)
	print(df['tag'].value_counts(normalize = True))
	print('Number of data sentences: {:,}\n'.format(df.shape[0]))

	df.to_csv(csv_file,index=False)