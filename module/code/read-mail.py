import os
import re
import email
import imaplib
import argparse
import pandas as pd

from email.header import decode_header
from bs4 import BeautifulSoup

import filter_en as fte
import utility as utl

def get_auth(cred):
	df = pd.read_csv(cred)
	return df['usr'][0], df['pwd'][0]

def read_imap(cred):
	# create an IMAP4 class with SSL 
	imap = imaplib.IMAP4_SSL("imap.gmail.com", '993')
	username, password = get_auth(cred)
	# authenticate
	imap.login(username, password)

	status, messages = imap.select("INBOX")
	# total number of emails
	N = int(messages[0])
	print('Total emails: {:}'.format(N))

	subjects = []
	froms = []
	bodys = []
	err = []

	for i in range(N, 0, -1):
		utl.show_bar(i, N, 50)
		if i<N-10:
			break

		# fetch the email message by ID
		res, msg = imap.fetch(str(i), "(RFC822)")
		for response in msg:
			try:
				if isinstance(response, tuple):
					# parse a bytes email into a message object
					msg = email.message_from_bytes(response[1])
					# decode the email subject
					subject = decode_header(msg["Subject"])[0][0]
					if isinstance(subject, bytes):
						subject = subject.decode()
					# email sender
					from_ = msg.get("From")

					from_, subject = clean_title(from_, subject)

					# if the email message is multipart
					if msg.is_multipart():
						multi = []
						for part in msg.walk():
							# extract content type of email
							content_type = part.get_content_type()
							content_disposition = str(part.get("Content-Disposition"))
							try:
								# get the email body
								body = part.get_payload(decode=True).decode()
							except:
								continue
							if content_type == "text/plain" and "attachment" not in content_disposition:
								multi.append(body)
						multi_str = '\r\n'.join(multi)
						bodys.append(multi_str)
					else:
						# extract content type of email
						content_type = msg.get_content_type()
						# get the email body
						body = msg.get_payload(decode=True).decode()
						if content_type == "text/plain":
							bodys.append(body)
						if content_type == "text/html":
							soup = BeautifulSoup(body, features="lxml")
							text = soup.get_text()
							bodys.append(text)
						else:
							continue

					froms.append(from_)
					subjects.append(subject)
			except Exception as e:
				err.append(index)
				print(e)
				continue

	print('\n\n')
	imap.close()
	imap.logout()

	return froms, subjects, bodys, err

def clean_title(from_, sub_):
	bad_chars = ['/','\\',':','*','?','"','<','>','|']
	for i in bad_chars :
		sub_ = sub_.replace(i,' ')
		from_ = from_.replace(i,' ')

	sub_ = re.sub(r'\s+',' ',sub_).strip()
	from_ = re.sub(r'\s+',' ',from_)
	return from_, sub_

def parser():
	ap = argparse.ArgumentParser()
	ap.add_argument("-c", "--cred", type=str, default=r'..\data\credentials.csv',
	help="path to input credentials csv file")
	ap.add_argument("-g", "--gmail", type=str, default=r'..\data\gmail.csv',
	help="path to output gmail csv file")
	ap.add_argument("-e", "--error", type=str, default=r'..\log\error.log',
	help="path to output error log")
	args = vars(ap.parse_args())
	return args

def validate(args):
	if os.path.isfile(args['cred']) and args['cred'].endswith('.csv'):
		pass
	else:
		return False, 'Credential must be CSV file'

	if not args['gmail'].endswith('.csv'):
		return False, 'Gmail output must be CSV file'

	utl.create_parent(args['gmail'])
	utl.create_parent(args['error'])
	return True, ''

if __name__ == '__main__':
	args = parser()

	credentials = args['cred']
	gmail_csv = args['gmail']
	err_path = args['error']

	v, m = validate(args)
	if v:
		froms, subjects, bodys, err = read_imap(credentials)

		utl.write_log(err_path, err)

		dict_ = {'From': froms, 'Sub':subjects, 'body':bodys}
		df = pd.DataFrame(dict_)

		df = fte.add_lang(df)
		en = fte.filter_en(df)
		df.drop(en.index, inplace=True)
		df.drop(['Lang'], axis=1, inplace=True)

		df.to_csv(gmail_csv, index=False)
	else:
		print('[ERROR] '.format(m))