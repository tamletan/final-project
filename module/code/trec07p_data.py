import os
import re
import email
import pandas as pd

from email.header import decode_header
from bs4 import BeautifulSoup

path_data = r'..\..\dataset\data-train\trec07p\data'
path_label = r'..\..\dataset\data-train\trec07p\full\index'

def replace_path(p):
	file = p.split('/')[-1]
	return os.path.join(path_data, file)

def get_text(path):
	try:
		with open(path, 'rb') as f:
			text = f.read()
			return text
	except Exception as e:
		print('Error: Cannot read')
		return '[ERROR]'

def clean_body(s):
	if not isinstance(s, str):
		s = str(s)
	s = re.sub(r'([\;\:\|•«])', ' ', s) 	# Remove special characters
	s = re.sub(r'(@.*?)[\s]', ' ', s) 		# Remove '@name'
	s = re.sub(r'&amp;', '&', s)  			# Replace '&amp;' with '&'
	s = re.sub('\n+','\n',s).strip()		# Replace multi newline
	return s

def extract(text):
	msg = email.message_from_bytes(text)

	if msg.is_multipart():
		multi = []
		for part in msg.walk():
			content_type = part.get_content_type()
			tmp = part.get_payload(decode=True)
			if type(tmp) is bytes:
				body = tmp.decode('latin-1')
				if content_type == "text/plain":
					multi.append(body)
				if content_type == "text/html":
					soup = BeautifulSoup(body, features="lxml")
					multi.append(soup.get_text())
		r = [clean_body(s) for s in multi]
		for i in r:
			tmp = re.sub(r'\s+',' ', i).strip()
			if tmp == '':
				continue
			return i
		
	else:
		content_type = msg.get_content_type()
		body = msg.get_payload(decode=True).decode('latin-1')

		if content_type == "text/plain":
			text = body
		if content_type == "text/html":
			soup = BeautifulSoup(body, features="lxml")
			text = soup.get_text()

		return clean_body(text)

def path_to_msg(s):
	path = replace_path(s)
	text = get_text(path)
	msg = extract(text)
	return msg

if __name__ == '__main__':
	df = pd.read_csv(path_label, sep=' ', names=['tag', 'body'])
	df['body'] = df['body'].apply(path_to_msg)
	print(df)
	df.to_csv('test_data.csv', index=False)

	# path = os.path.join(path_data,'inmail.22')
	# text = get_text(path)
	# body = extract(text)
	# print(type(body))
	# print(body)

	df = pd.read_csv('test_data.csv')
	print(f"Number of rows: {df.shape[0]:,}\n")
	print(df["tag"].value_counts(normalize = True))

	print("\nDrop {:,} row with null value\n".format(df["body"].isnull().sum()))
	df.dropna(subset=["body"], inplace=True)
	print(f"Number of remain rows: {df.shape[0]:,}\n")

	df.to_csv('clean.csv', index=False)
