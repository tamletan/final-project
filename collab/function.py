import email
import imaplib
from bs4 import BeautifulSoup
from predictor import Predictor

def function(username, password):
	imap = imaplib.IMAP4_SSL("imap.gmail.com", "993")
	imap.login(username, password)
	imap.select("INBOX")
	# list unread mail
	status, messages = imap.search(None, "(UNSEEN)")
	unread = messages[0].decode().split()
	print(f'Total mails: {len(unread)}')

	procedure(imap, unread)

	print('\n\n')
	imap.close()
	imap.logout()

def procedure(imap, unread):
	predictor = Predictor()
	for mail in unread:
		# fetch the email message by ID
		res, msg = imap.fetch(mail, "(RFC822)")
		bodys = []
		for response in msg:
			try:
				if isinstance(response, tuple):
					# parse a bytes email into a message object
					msg = email.message_from_bytes(response[1])

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
						text = '\r\n'.join(multi)
					else:
						# extract content type of email
						content_type = msg.get_content_type()
						# get the email body
						body = msg.get_payload(decode=True).decode()
						if content_type == "text/plain":
							text = body
						if content_type == "text/html":
							soup = BeautifulSoup(body, features="lxml")
							text = soup.get_text()
						else:
							continue

					bodys.append(text)
				else:
					raise
			except:
				print("[ERROR] ",mail)
				imap.store(mail, "-FLAGS", "\\Seen")
				continue

		if not bodys:
			imap.store(mail, "-FLAGS", "\\Seen")
		else:
			tag = predictor.predict(bodys[0])
			if tag:
				# mark as unread
				# to mark as read , replace -FLAGS to +FLAGS
				imap.store(mail, "-FLAGS", "\\Seen")
				print("[Seen] ",mail)
			else:
				# delete mail
				imap.store(mail, "+X-GM-LABELS", "\\Trash")
				print("[Trash] ",mail)

if __name__ == '__main__':
	function(usr, pwd)