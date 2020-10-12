import imaplib
import email
import codecs
from email.header import decode_header
import webbrowser
import os
from bs4 import BeautifulSoup

# account credentials
username = "tamletannk94@gmail.com"
password = "wjstsopvcnhfjjvv"

# create an IMAP4 class with SSL 
imap = imaplib.IMAP4_SSL("imap.gmail.com", '993')
# authenticate
imap.login(username, password)

status, messages = imap.select("INBOX")
# number of top emails to fetch
N = 1
# total number of emails
messages = int(messages[0])

for i in range(messages, messages-N, -1):
    # fetch the email message by ID
    res, msg = imap.fetch(str(i), "(RFC822)")
    for response in msg:
        if isinstance(response, tuple):
            # parse a bytes email into a message object
            msg = email.message_from_bytes(response[1])
            print(msg)
            break
            # decode the email subject
            subject = decode_header(msg["Subject"])[0][0]
            if isinstance(subject, bytes):
                # if it's a bytes, decode to str
                subject = subject.decode()
            # email sender
            from_ = msg.get("From")
            print("Subject:", subject)
            print("From:", from_)
            # if the email message is multipart
            if msg.is_multipart():
                # iterate over email parts
                for part in msg.walk():
                    # extract content type of email
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition"))
                    try:
                        # get the email body
                        body = part.get_payload(decode=True).decode()
                    except:
                        pass
                    if content_type == "text/plain" and "attachment" not in content_disposition:
                        # print text/plain emails and skip attachments
                        print(body)
                    elif "attachment" in content_disposition:
                        # download attachment
                        filename = part.get_filename()
                        if filename:
                            if not os.path.isdir(subject):
                                # make a folder for this email (named after the subject)
                                os.mkdir(subject)
                            filepath = os.path.join(subject, filename)
                            # download attachment and save it
                            open(filepath, "wb").write(part.get_payload(decode=True))
            else:
                # extract content type of email
                content_type = msg.get_content_type()
                # get the email body
                body = msg.get_payload(decode=True).decode()
                if content_type == "text/plain":
                    # print only text email parts
                    print(body)
            if content_type == "text/html":
                bad_chars = ['/','\\',':','*','?','"','<','>','|']
                for i in bad_chars : 
                    subject = subject.replace(i,'') 
                # if it's HTML, create a new HTML file and open it in browser
                if not os.path.isdir(subject):
                    # make a folder for this email (named after the subject)
                    os.mkdir(subject)
                filename = f"{subject[:50]}.html"
                filepath = os.path.join(subject, filename)
                soup = BeautifulSoup(body, features="lxml")
                print(soup.get_text())
                # write the file
                open(filepath, "w").write(str(soup.get_text().encode('ascii', 'xmlcharrefreplace')))
                # open(filepath, "w").write(str(strin.encode('ascii', 'xmlcharrefreplace')))
                # open in the default browser
                webbrowser.open(filepath)
            # print("="*100)

imap.close()
imap.logout()