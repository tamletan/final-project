from flask import Flask, render_template, url_for, request

import sys
sys.path.insert(0, '../code')
import torch
import pandas as pd
from utils import *
from constants import *

def import_model():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = init_model()
	model = model.to(device)
	model.load_state_dict(torch.load('../code'/data_path/weight_file))
	tokenizer = BertTokenizerFast.from_pretrained(bert_pretrain)
	return model, tokenizer, device

def predict(content, model, tokenizer, device):
	content = clean_content(content)
	if content == '':
		return 1
	text = np.asarray([content])
	label = np.asarray([1])
	text_token = tokens(tokenizer, text, 256)
	data_loader = create_loader(text_token, label, 32, False)
	predictions = predict_model(device, model, data_loader)
	return predictions[0]

app = Flask(__name__)
result = ''
model, tokenizer, device = import_model()

@app.route('/', methods=['POST','GET'])
def index():
	return render_template('index.html', result = result )

@app.route('/onClick', methods=['POST','GET'])
def onClick():
	if request.method == 'POST':
		content = request.form['content']
		result = predict(content, model, tokenizer, device)
		return render_template('index.html', result = 'Spam' if result==1 else 'Legit')
	return render_template('index.html', result = '')

if __name__ == "__main__":
	app.run(debug=True)