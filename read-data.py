import pandas as pd

import os
# print(os.listdir("dataset"))

# dataset = pd.read_csv(r'dataset/emails.csv')
# print(dataset.columns)
# print(dataset.shape)
# dataset.drop_duplicates(inplace = True)
# print(dataset.shape)

# print (pd.DataFrame(dataset.isnull().sum()))
# text = [sub for sub in body.split('\n') if ':' not in sub]
# res = dict(map(str.strip, sub.split(':', 1)) for sub in body.split('\n') if ':' in sub) 

err = []
chunksize = 10 ** 5
textplain = 0
pack = 0
textpath = r'.\dataset\data-extract'
for chunk in pd.read_csv(r'dataset\emails.csv', chunksize=chunksize):
	rows = chunk['message'].to_dict()
	for row in rows:
		msg = rows[row]
		body = msg

		dirpath = os.path.join(textpath,str(pack))
		if not os.path.exists(dirpath):
			os.makedirs(dirpath)
		filepath = os.path.join(dirpath,str(textplain))

		for sub in msg.split('\n'):
			if "X-FileName:" in sub:
				body = body.replace(sub+'\n','')
				break
			body = body.replace(sub+'\n','')

		textplain += 1

		f = open(filepath, "w")
		try:
			f.write(body)
			print('{:<6}'.format(textplain), end='\r')
			# print('=', end='')
		except:
			err.append(textplain-1)
			f.write(str(body.encode('ascii', 'xmlcharrefreplace')))

	pack += 1

print('\nError: ',err)