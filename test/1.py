import pandas as pd
import os
import re

from email.parser import Parser
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import FreqDist

root = r".\data"
dataset = r"..\dataset\emails.csv"
chunksize = 10 ** 3
to_adr = []
from_adr = []
contents = []


def read_data(source, csize):
	for chunk in pd.read_csv(source, chunksize=csize):
		rows = chunk['message'].to_dict()
		for i in rows:
			msg = rows[i]
			email = Parser().parsestr(msg)

			to_adr.append(adr_format(email['to']))
			from_adr.append(adr_format(email['from']))
			contents.append(email.get_payload().strip())
		break

def store_data(to_path, from_path, content_path):
	to_file = os.path.join(root, to_path)
	from_file = os.path.join(root, from_path)
	body_file = os.path.join(root, content_path)

	with open(to_file, "w") as f:
		for to in to_adr:
			if to:
				try:
					f.write(to)
					f.write('\n')
				except:
					print("To:",to)

	with open(from_file, "w") as f:
		for fr in from_adr:
			if fr:
				try:
					f.write(fr)
					f.write('\n')
				except:
					print("From:",fr)

	with open(body_file, "w") as f:
		for body in contents:
			if body:
				try:
					f.write(body)
					f.write('\n'+'='*200+'\n')
				except:
					print("Body:",body)

def adr_format(arr):
	if arr:
		return re.sub(r"([^\w@.,])",'',arr)

def show(obj):
	for i in obj:
		print(format(i[0],'15s'),':',format(i[1],'6d'))

if __name__ == '__main__':
	read_data(dataset, chunksize)
	store_data("to.txt","from.txt","body.txt")

	with open(os.path.join(root,"body.txt"), "r") as f:
		data = f.read()
		data = data.replace('=','')
		words= word_tokenize(data)
 
		useful_words = [word  for word in words if word not in stopwords.words('English')]
		frequency = FreqDist(useful_words)

		show(frequency.most_common(100))