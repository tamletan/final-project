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

def get_auth(cred):
	df = pd.read_csv(cred)
	return df['usr'][0], df['pwd'][0]

imap = imaplib.IMAP4_SSL("imap.gmail.com", '993')
username, password = get_auth(r'..\data\credentials.csv')
imap.login(username, password)
status, messages = imap.select("INBOX")
N = int(messages[0])
print('Total emails: {:}'.format(N))

for i in range(N-5, 0, -1):
	res, msg = imap.fetch(str(i), "(RFC822)")
	for response in msg:
		if isinstance(response, tuple):
			msg = email.message_from_bytes(response[1])
			subject = decode_header(msg["Subject"])[0][0]
			if isinstance(subject, bytes):
				subject = subject.decode()
			# email sender
			from_ = msg.get("From")
			print(from_)
			print(subject)
			# imap.store(str(i), '+X-GM-LABELS', '\\Trash')
		break
	break

imap.expunge()
imap.close()
imap.logout()