import os
import re
from sys import stdout

def create_parent(child_path):
	parent = os.path.sep.join(child_path.split(os.path.sep)[0:-1])
	if not os.path.exists(parent):
		os.makedirs(parent)

def show_bar(i, count, size):
	x = int(size * i / count)
	stdout.write("{:} [{:}{:}] {:}/{:}\r".format('Email', '#'*(size-x), '.'*x, i, count))
	stdout.flush()

def write_log(path, data):
	create_parent(path)
	try:
		with open(path, 'w') as f:
			value = '\n'.join(data)
			f.write(value)
	except Exception as e:
		print('Error: Cannot write')
		print(e)

def read_log(path):
	if os.path.isfile(path):
		try:
			f = open(path, 'r')
			text = f.read()
			data = text.split('\n')
			return data
		except Exception as e:
			print('Error: Cannot read file')
			return []
	else:
		print('Error: File not exist')
		return []