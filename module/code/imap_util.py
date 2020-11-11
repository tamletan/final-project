import os
import re
import email
import imaplib
import argparse
import pandas as pd
import time

from email.header import decode_header
from bs4 import BeautifulSoup

import filter_en as fte
import utility as utl

def parser():
	ap = argparse.ArgumentParser()
	ap.add_argument("-c", "--cred", type=str, default=r'..\data\credentials.csv',
	help="path to input credentials csv file")
	args = vars(ap.parse_args())
	return args

def read_imap(cred):
	imap = open_imap(cred)
	status, messages = imap.select("INBOX")
	# total number of emails
	N = int(messages[0])
	print('Total emails: {:}'.format(N))
	index = N-1
	print(f'Index emails: {index}')

	res, msg = imap.fetch(str(index), "(RFC822)")
	print(msg[0][0].decode())
	for response in msg:
		try:
			if isinstance(response, tuple):
				# parse a bytes email into a message object
				msg = email.message_from_bytes(response[1])
				get_header(msg)
				mark_unread(str(index))
				# move_trash(str(index))
		except:
			continue
	close_imap(imap)

def open_imap(cred):
	# create an IMAP4 class with SSL 
	imap = imaplib.IMAP4_SSL("imap.gmail.com", '993')
	username, password = utl.get_auth(cred)
	# authenticate
	imap.login(username, password)
	return imap

def close_imap(imap):
	imap.expunge()
	imap.close()
	imap.logout()

def get_header(msg):
	subject = decode_header(msg["Subject"])[0][0]
	if isinstance(subject, bytes):
		subject = subject.decode()
	from_ = msg['From']
	uid = msg['Message-ID']
	print(f'From    : {from_}\nSubject : {subject}\nUID     : {uid}')

def move_trash(imap, uid):
	imap.store(uid, '+X-GM-LABELS', '\\Trash')

def mark_read(imap, uid):
	imap.store(uid, '+FLAGS', '\\Seen')

def mark_unread(imap, uid):
	imap.store(uid, '-FLAGS', '\\Seen')

if __name__ == '__main__':
	args = parser()
	read_imap(args['cred'])
