#!/usr/bin/python3.6
"""AutoDelete
This script auto deletes messages from your gmail inbox.
The perfect usecase is deleting the OTP messages received from banks after certain time period.
or deleting the messages received from certain services which doesn't have unsubscribe option.
Many parts of this script are copied from EZGmail by Al Sweigart (https://github.com/asweigart/ezgmail)
"""

import os
import logging
import base64
import email

from googleapiclient.discovery import build
from httplib2 import Http
from oauth2client import file, client, tools
from apiclient import errors

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
file_handler = logging.FileHandler('u_auto_delete_it.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

SCOPES = 'https://mail.google.com/'  # read-write mode
SERVICE = None

class AutoDeleteException(Exception):
	pass  # This class exists for this module to raise for AutoDelete-specific problems.


def init(user_id='me', token_file='token.json', credentials_file='credentials.json'):
	"""Populates SERVICE  global variable
	The account is determined by the credentials.json file, downloaded from Google, and token.json. If the token.json file hasn't been generated yet, this function will open the browser to a page to let the user log in to the Gmail account that he will use.
	Parameters
	__________
	user_id : str
		User id of the Gmail User (default is 'me')
	token_file : str
		The filename of 'token.json' file (This is auto downloaded when this function runs)
	credentials_file : str
		The filename of 'credentials.json. This can be downloaded from 'https://developers.google.com/gmail/api/quickstart/python and clicking "Enable the Gmail API'
	Returns
	_______
	SERVICE
	"""

	global SERVICE

	if not os.path.exists(credentials_file):
		raise AutoDeleteException('Can\'t find credentials file at %s. You can download this file from https://developers.google.com/gmail/api/quickstart/python and clicking "Enable the Gmail API"' % (os.path.abspath(credentials_file)))

	store = file.Storage(token_file)
	creds = store.get()
	if not creds or creds.invalid:
		flow = client.flow_from_clientsecrets(credentials_file, SCOPES)
		creds = tools.run_flow(flow, store)
	SERVICE = build('gmail', 'v1', http=creds.authorize(Http()))


def search(query, user_id='me'):
	"""Returns a list of gmail thread objects of all the messages matching query
	Parameters
	__________
	query : str
		The string Exactly you will use in Gmail's Search Box
		label:UNREAD
		from:abc@email.com
		subject:hello
		has:attachment
		more described at https://support.google.com/mail/answer/7190?hl=en
	user_id : str
		User id of the Gmail User (default is 'me')
	Returns
	_______
	list
		List of Messages that match the criteria of the query.
		Note that the returned list contains Message IDs, you must use get with the appropriate ID to get the details of a Message.
	"""

	if SERVICE is None:
		init()

	try:
		response = SERVICE.users().messages().list(userId=user_id,
												   q=query).execute()

		messages = []
		if 'messages' in response:
			messages.extend(response['messages'])

		while 'nextPageToken' in response:
			page_token = response['nextPageToken']
			response = SERVICE.users().messages().list(userId=user_id, q=query,
													   pageToken=page_token).execute()
			messages.extend(response['messages'])

		return messages
		# print(len(messages))

	except errors.HttpError as e:
		logger.exception(f'An error occurred:{e}')


def delete_messages(query, user_id='me'):
	"""Deletes the message matching the query
	Parameters
	__________
	query : str
		The string Exactly you will use in Gmail's Search Box
		label:UNREAD
		from:username@email.com
		subject:hello
		has:attachment
		more described at https://support.google.com/mail/answer/7190?hl=en
	user_id : str
		User id of the Gmail User (default is 'me')
	"""
	messages = search(query)
	if messages:
		for message in messages:
			SERVICE.users().messages().delete(userId=user_id, id=message['id']).execute()
			logger.info(f'Message with id: {message["id"]} deleted successfully.')
	else:
		logger.info("There was no message matching the query.")
		
def read(query, user_id='me'):
	if SERVICE is None:
		init()

	try:
		response = SERVICE.users().messages().list(userId=user_id, q=query).execute()
		messages = []

		count = 0;
		
		contents = response.get('messages', [])
		contents.reverse()
		if not contents:
			print ("No messages found.")
		else:
			print ("Message contents:")
			for message in contents:
				msg = SERVICE.users().messages().get(userId='me', id=message['id'], format='full').execute()
				print(msg)
				# data = list(findkeys(msg, 'data'))
				# # data = msg['payload']['parts'][0]['body']['data']
				# msg_str = base64.urlsafe_b64decode(data[0].encode("ASCII")).decode("utf-8")
				# mime_msg = email.message_from_string(msg_str)
				# messages.append(mime_msg)
				break

		return messages

	except errors.HttpError as e:
		logger.exception(f'An error occurred:{e}')

def findkeys(node, kv):
	if isinstance(node, list):
		for i in node:
			for x in findkeys(i, kv):
			   yield x
	elif isinstance(node, dict):
		
		if 'body' in node:
			if node['mimeType'] == 'text/plain':
				if kv in node['body']:
					yield node['body'][kv]

		for j in node.values():
			for x in findkeys(j, kv):
				yield x

if __name__ == '__main__':
	# logger.info("Deleting messages from tamletannk94@gmail.com.")

	# all email
	query = 'ntxly'
	# query = 'adrian'
	s = read(query)
	print(s[0].as_string()[:500],'...')