import os
import re
import email
import imaplib
import pandas as pd
from email.header import decode_header
from bs4 import BeautifulSoup

def get_auth(cred):
	df = pd.read_csv(cred)
	return df['usr'][0], df['pwd'][0]

def clean_title(from_, sub_):
	for i in bad_chars :
		sub_ = sub_.replace(i,' ')
		from_ = from_.replace(i,' ')

	sub_ = re.sub(r'\s+',' ',sub_).strip()
	from_ = re.sub(r'\s+',' ',from_)
	return from_, sub_

def read_imap(cred):
	# create an IMAP4 class with SSL 
	imap = imaplib.IMAP4_SSL("imap.gmail.com", '993')
	username, password = get_auth(cred)
	# authenticate
	imap.login(username, password)

	status, messages = imap.select("INBOX")
	# total number of emails
	messages = int(messages[0])
	# number of top emails to fetch
	N = messages
	print('Total emails: {:}'.format(messages))

	subjects = []
	froms = []
	bodys = []
	err = []
	bad_chars = ['/','\\',':','*','?','"','<','>','|']

	for i in range(messages, messages-N, -1):
		index = str(i)
		print(index)
		# fetch the email message by ID
		res, msg = imap.fetch(index, "(RFC822)")
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
			except:
				err.append(index)
				continue

	imap.close()
	imap.logout()
	with open(r'..\log\error.txt', 'w') as f:
		value = '\n'.join(err)
		f.write(value)
	print('Err: ', err)
	return froms, subjects, bodys

if __name__ == '__main__':
	credentials = r'..\data\credentials.csv'
	gmail_csv = r'..\data\gmail.csv'

	froms, subjects, bodys = read_imap(credentials)

	dict_ = {'From': froms, 'Sub':subjects, 'body':bodys}
	df = pd.DataFrame(dict_)
	df.to_csv(gmail_csv, index=False)


