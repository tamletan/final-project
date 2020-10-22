import re
import os
import pandas as pd

# corpus = 'stop'
# root = r'..\dataset\data-train\lingspam_public\{}'.format(corpus)
# save_csv = r'.\{}.csv'.format(corpus)
root = r'..\dataset\data-train\corpus'
save_csv = r'.\corpus.csv'
re_exp = r'.*spmsg.*'

def get_file(dataset_path, expression):
	f = []
	for dirs, _, files in os.walk(dataset_path):
		if files:
			print(dirs)
			for file in files:
				f.append(os.path.join(dirs,file))

	r = re.compile(expression)
	sp = list(filter(r.match, f))
	lg = list(filter(lambda m: m not in sp, f))

	return sp, lg

def read_file(dataset_path, expression):
	spam, legit = get_file(dataset_path, expression)

	tags = []
	bodys = []

	for file in spam:
		with open(file, 'r') as f:
			data = f.read()
			body = re.sub('Subject:.*\n\n', '', data)
			bodys.append(body)
			tags.append('spam')

	for file in legit:
		with open(file, 'r') as f:
			data = f.read()
			body = re.sub('Subject:.*\n\n', '', data)
			bodys.append(body)
			tags.append('legit')

	data = {'tag': tags, 'body': bodys}
	return data

if __name__ == '__main__':
	data = read_file(root, re_exp)

	df = pd.DataFrame(data, columns = ['tag', 'body'])
	print(df['tag'].value_counts())
	df.to_csv(save_csv,index=False)