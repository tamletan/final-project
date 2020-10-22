import os
import re

from email.parser import Parser
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import FreqDist

def read_person(person):
	rootdir = os.path.join(dataset,person)

	# for directory, subdirectory, filenames in  os.walk(rootdir):
	#     print(directory, subdirectory, len(filenames))

	filedir = os.path.join(rootdir,r"_sent\1")

	with open(filedir, "r") as f:
		data = f.read()
		email = Parser().parsestr(data)

		print("To: " , email['to'])
		print("From: " , email['from'])
		print("\nBody: " , email.get_payload())

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

def get_file(dataset):
	f = []
	count = 0
	for dirs, _, files in os.walk(dataset):
		if count !=2:
			count+=1
			continue

		if files:
			print(dirs)
			for file in files:
				f.append(os.path.join(dirs,file))

		count += 1
		if count==2:
			break
	return f

def adr_format(arr):
	if arr:
		return re.sub(r"([^\w@.,])",'',arr)

def show(obj):
	for i in obj:
		print(format(i[0],'15s'),':',format(i[1],'6d'))

if __name__ == '__main__':
	dataset = r"..\dataset\enron_mail\maildir"
	to_adr = []
	from_adr = []
	contents = ""

	# read_person("lay-k")
	files = get_file(dataset)
	for file in files:
		with open(file,"r") as f:
			data = f.read()
			email = Parser().parsestr(data)
			to_adr.append(adr_format(email['to']))
			from_adr.append(adr_format(email['from']))
			contents+=email.get_payload().strip()

	words= word_tokenize(contents)
	useful_words = [word  for word in words if word not in stopwords.words('English')]
	frequency = FreqDist(useful_words)

	show(frequency.most_common(100))