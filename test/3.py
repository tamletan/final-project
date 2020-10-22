import re
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 

root = r'..\dataset\data-train\lingspam_public\stop'
re_exp = r'.*spmsg.*'

def get_file(dataset_path, expression):
	f = []
	for dirs, _, files in os.walk(dataset_path):
		if files:
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

def proc(df):
	print(df['tag'].value_counts())
	df_spam = df[df.tag=='spam']
	spam_list = df_spam['body'].tolist()
	filtered_spam = ("").join(spam_list)
	filtered_spam = filtered_spam.lower()
	print(filtered_spam[:200])

def split_data(df):
	df['tag'] = df['tag'].apply(lambda x: 1 if x == 'spam' else 0)

	x_train, x_test, y_train, y_test = train_test_split(df['body'], df['tag'], test_size = 0.3, random_state = 0)

	print('rows in test set: ' + str(x_test.shape))
	print('rows in train set: ' + str(x_train.shape))
	print(type(x_train))

	return x_train, x_test, y_train, y_test

def tfidf(df):
	x_train, x_test, y_train, y_test = split_data(df)
	list = x_train.tolist()
	vectorizer = TfidfVectorizer(
		input= list ,  			# input is actual text
		lowercase=True,      	# convert to lower case before tokenizing
		stop_words='english' 	# remove stop words
	)
	features_train_transformed = vectorizer.fit_transform(list) 	#gives tf idf vector
	features_test_transformed  = vectorizer.transform(x_test) 		#gives tf idf vector

	classifier = MultinomialNB()
	classifier.fit(features_train_transformed, y_train)
	print("classifier accuracy {:.2f}%".format(classifier.score(features_test_transformed, y_test) * 100))

	labels = classifier.predict(features_test_transformed)

	actual = y_test.tolist() 
	predicted = labels 
	results = confusion_matrix(actual, predicted) 

	print('Confusion Matrix :')
	print(results) 
	print ('Accuracy Score :',accuracy_score(actual, predicted)) 
	print ('Report : ')
	print (classification_report(actual, predicted) )

	score_2 = f1_score(actual, predicted, average = 'binary')
	print('F-Measure: %.3f' % score_2)

if __name__ == '__main__':
	data = read_file(root, re_exp)

	df = pd.DataFrame(data, columns = ['tag', 'body'])
	print(df['tag'].value_counts())

	tfidf(df)

	# proc(df)