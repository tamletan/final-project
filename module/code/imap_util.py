import os
import re
import email
import imaplib
import argparse
import pandas as pd
import time

from email.header import decode_header

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
	
	for i in range(N, N-50, -1):
		index = str(i)
		res, msg = imap.fetch(index, "(RFC822)")
		print(index)
		# print(msg[0][0].decode())
		for response in msg:
			try:
				if isinstance(response, tuple):
					# parse a bytes email into a message object
					msg = email.message_from_bytes(response[1])
					# get_header(msg)
			except:
				continue
		# move_trash(imap, index)
		mark_unread(imap, index)

	close_imap(imap)

def open_imap(cred):
	# create an IMAP4 class with SSL 
	imap = imaplib.IMAP4_SSL("imap.gmail.com", '993')
	username, password = utl.get_auth(cred)
	# authenticate
	imap.login(username, password)
	status, messages = imap.select("INBOX")
	N = int(messages[0])
	print('Total emails: ',N)
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

def read_unseen(imap):
	retcode, messages = imap.search(None, '(UNSEEN)')
	if retcode=='OK':
		msg = messages[0].split()
		print('Total unseen: ',len(msg))
		return msg

def delete_by_id(imap, uid):
	for index in uid:
		mark_read(imap, index)
		# move_trash(imap, index)
		print('Delete mail:',index)

if __name__ == '__main__':
	args = parser()
	ids = ['1650', '1640']
	imap = open_imap(args['cred'])
	read_unseen(imap)
	delete_by_id(imap, ids)
	read_unseen(imap)
	close_imap(imap)
