import imaplib
import email
import codecs
from email.header import decode_header
import os
from bs4 import BeautifulSoup
import pandas as pd

bad_chars = ['/','\\',':','*','?','"','<','>','|','\r\n']

# account credentials
username = "tamletannk94@gmail.com"
password = "krarddhmjrqljkcw"

def read_imap():
	# create an IMAP4 class with SSL 
	imap = imaplib.IMAP4_SSL("imap.gmail.com", '993')
	# authenticate
	imap.login(username, password)

	status, messages = imap.select("INBOX")
	# total number of emails
	messages = int(messages[0])
	# number of top emails to fetch
	N = messages
	print('Total emails: {:}'.format(messages))
	'''
	res, msg = imap.search(None, '(FROM "ntxly@dut.udn.vn")')
	list_ = str(msg[0])
	list_ = list_.replace("b'",'')
	list_ = list_.replace("'",'')
	arr_ = list_.split(' ')
	print(arr_)
	res, msg = imap.fetch(arr_[3], "(RFC822)")
	for response in msg:
		if isinstance(response, tuple):
			msg = email.message_from_bytes(response[1])
			subject = decode_header(msg["Subject"])[0][0]
			if isinstance(subject, bytes):
				# if it's a bytes, decode to str
				subject = subject.decode()
			print("Subject:", subject)
	'''
	subjects = []
	froms = []
	bodys = []

	for i in range(messages, messages-N, -1):
		index = os.path.join('mail', str(i))
		print(index)
		# fetch the email message by ID
		res, msg = imap.fetch(str(i), "(RFC822)")
		for response in msg:
			if isinstance(response, tuple):
				# parse a bytes email into a message object
				msg = email.message_from_bytes(response[1])
				# decode the email subject
				subject = decode_header(msg["Subject"])[0][0]
				if isinstance(subject, bytes):
					# if it's a bytes, decode to str
					subject = subject.decode()
				# email sender
				from_ = msg.get("From")
				# print("Subject:", subject)
				# print("From:", from_)

				for i in bad_chars :
					subject = subject.replace(i,'')
					from_ = from_.replace(i,'')
				subject = subject.strip()

				# if not os.path.isdir(index):
				# 	os.makedirs(index)

				# if the email message is multipart
				if msg.is_multipart():
					multi = []
					# iterate over email parts
					for part in msg.walk():
						# extract content type of email
						content_type = part.get_content_type()
						content_disposition = str(part.get("Content-Disposition"))
						try:
							# get the email body
							body = part.get_payload(decode=True).decode()
							raw_body = part.get_payload(decode=True)
						except:
							pass
						if content_type == "text/plain" and "attachment" not in content_disposition:
							# # print text/plain emails and skip attachments
							# filepath = os.path.join(index, subject[:50]+'.txt')
							# # download attachment and save it
							# open(filepath, "wb").write(raw_body)
							multi.append(body)
							# print("==== text/plain ==== (not attachment) \n")
							# print(body)
						elif "attachment" in content_disposition:
							# download attachment
							filename = part.get_filename()
							# if filename:
							# 	# filepath = os.path.join(index, filename)
							# 	# # download attachment and save it
							# 	# open(filepath, "wb").write(raw_body)
							# 	# multi.append(raw_body)
							# 	print("==== text/plain ==== (attachment) \n")
							# 	# print(body)
					multi_str = '\r\n'.join(multi)
					bodys.append(multi_str)
					froms.append(from_)
					subjects.append(subject)
				else:
					# extract content type of email
					content_type = msg.get_content_type()
					try:
						# get the email body
						body = msg.get_payload(decode=True).decode()
						raw_body = part.get_payload(decode=True)
					except:
						pass
					if content_type == "text/plain":
						bodys.append(body)
						froms.append(from_)
						subjects.append(subject)
						# print only text email parts
						# print("==== text/plain ==== \n")
						# print(body)
					if content_type == "text/html":
						# # if it's HTML, create a new HTML file and open it in browser
						# filename = f"{subject[:50]}.html.txt"
						# filepath = os.path.join(index, filename)
						soup = BeautifulSoup(body, features="lxml")
						# print("==== text/html  ==== \n")
						text = soup.get_text()
						bodys.append(text)
						froms.append(from_)
						subjects.append(subject)
						# # write the file
						# f = open(filepath, "wb")
						# try:
						# 	f.write(text.encode())
						# except:
						# 	print('Cannot write')

	imap.close()
	imap.logout()
	return froms, subjects, bodys

if __name__ == '__main__':
	froms, subjects, bodys = read_imap()
	dict_ = {'From': froms, 'Sub':subjects, 'Body':bodys}
	df = pd.DataFrame(dict_)
	df.to_csv('tamletannk94.csv', index=False)

	# dt = {'From': str, 'Sub': str, 'Body': bytes}
	df = pd.read_csv('tamletannk94.csv')
	# # # print(df['Sub'])
	print(df[['Sub','Body']].head(5))
	print(df[['Sub','Body']].tail(5))
	# print(df['Body'][5])

