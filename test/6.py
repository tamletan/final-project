import os
import re
import pandas as pd

def clear_sub1(s):
	if not isinstance(s,str):
	  s = str(s)
	s = re.sub('Subject:', '', s)
	return s

def clear_sub2(s):
	if not isinstance(s,str):
	  s = str(s)
	s = re.sub('Subject:.*\n', '', s)
	return s

def read_file1(path):
	df = pd.read_csv(path)
	df.rename({"text": "body", "spam": "tag"},axis=1, inplace=True)
	df['body'] = df['body'].apply(clear_sub1)
	return df['tag'], df['body']

def read_file2(path):
	df = pd.read_csv(path)
	df.drop(["Unnamed: 0", "label"], inplace=True, axis=1)
	df.rename({"text": "body", "label_num": "tag"},axis=1, inplace=True)
	df['body'] = df['body'].apply(clear_sub2)
	return df['tag'], df['body']

def read_file3(path):
	df = pd.read_csv(path, encoding='latin-1')
	df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], inplace=True, axis=1)
	df["v1"].replace({"ham": 0, "spam":1}, inplace=True)
	df.rename({"v1": "tag", "v2": "body"},axis=1, inplace=True)
	return df['tag'], df['body']

def merge_file(root1, root2, root3):
	tag1, body1 = read_file1(root1)
	tag2, body2 = read_file2(root2)
	tag3, body3 = read_file3(root3)

	tags = tag1.append(tag2, ignore_index = True)
	tags = tags.append(tag3, ignore_index = True)
	bodys = body1.append(body2, ignore_index = True)
	bodys = bodys.append(body3, ignore_index = True)

	return tags, bodys

def save_file(tags, bodys, csv_file):
	frame = { 'tag': tags, 'body': bodys }
	df = pd.DataFrame(frame)
	df["tag"].replace({0 : "legit", 1 : "spam"}, inplace=True)

	print(df['tag'].value_counts(normalize = True))

	df.to_csv(csv_file, index=False)

if __name__ == '__main__':
	root1 = r'..\dataset\data-train\1.csv'
	root2 = r'..\dataset\data-train\2.csv'
	root3 = r'..\dataset\data-train\3.csv'
	csv_file = r'.\set3.csv'	
	tags, bodys = merge_file(root1, root2, root3)
	save_file(tags, bodys, csv_file)
