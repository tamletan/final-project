from constants import *
import torch
import numpy as np
from format_data import tokens
from transformers import BertTokenizerFast

class Predictor():
	def __init__(self):
		super(Predictor, self).__init__()

		self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

		self.model = torch.load(data_path/model_file)
		self.model = self.model.to(self.device)
		self.model.load_state_dict(torch.load(data_path/weight_file))

		self.tokenizer = BertTokenizerFast.from_pretrained(bert_pretrain)

	def predict(self, text):
		token = tokens(self.tokenizer, np.asarray([text]), 256)
		seq = torch.tensor(token['input_ids'])
		mask = torch.tensor(token['attention_mask'])

		seq = seq.to(self.device)
		mask = mask.to(self.device)

		with torch.no_grad():
			preds = self.model(seq, mask)
			preds = preds.detach().cpu().numpy()

		predictions = np.argmax(preds)
		return True if predictions==0 else False