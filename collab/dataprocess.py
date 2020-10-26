import torch
import torch.nn as nn
import numpy as np
from timeit import default_timer as timer
from transformers import AutoModel, BertTokenizerFast, AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

def split_data(body, tag):
	state = 2018
	start = timer()
	print(format("Split data", '18s'), end='...')

	train_text, temp_text, train_labels, temp_labels = train_test_split(
		body, tag,
		random_state = state,
		test_size = 0.3,
		stratify = tag)

	# use temp set to create validation and test set
	val_text, test_text, val_labels, test_labels = train_test_split(
		temp_text, temp_labels,
		random_state = state,
		test_size = 0.5,
		stratify = temp_labels)

	print(" Elapsed time: {:.3f}".format(timer()-start))
	return train_text, train_labels, val_text, val_labels, test_text, test_labels

def tokens(tokenizer, data, MAX_LEN):
	t = tokenizer.batch_encode_plus(
		data.tolist(),
		max_length = MAX_LEN,
		padding=True,
		truncation=True,
		return_token_type_ids=False
	)
	return t

def tokenize(tokenizer, train_text, val_text, test_text, MAX_LEN):
	start = timer()
	print(format("Tokenize", '18s'), end='...')

	# tokenize and encode sequences
	tokens_train = tokens(tokenizer, train_text, MAX_LEN)
	tokens_val = tokens(tokenizer, val_text, MAX_LEN)
	tokens_test = tokens(tokenizer, test_text, MAX_LEN)

	print(" Elapsed time: {:.3f}".format(timer()-start))
	return tokens_train, tokens_val, tokens_test

def data_loader(tokens_train, train_labels, tokens_val, val_labels, tokens_test, test_labels, batch_size):
	start = timer()
	print(format("Seq to Tensor", '18s'), end='...')

	# for train set
	train_seq = torch.tensor(tokens_train['input_ids'])
	train_mask = torch.tensor(tokens_train['attention_mask'])
	train_y = torch.tensor(train_labels.tolist())

	# for validation set
	val_seq = torch.tensor(tokens_val['input_ids'])
	val_mask = torch.tensor(tokens_val['attention_mask'])
	val_y = torch.tensor(val_labels.tolist())

	# for test set
	test_seq = torch.tensor(tokens_test['input_ids'])
	test_mask = torch.tensor(tokens_test['attention_mask'])
	test_y = torch.tensor(test_labels.tolist())

	print(" Elapsed time: {:.3f}".format(timer()-start))

	start = timer()
	print(format("Create DataLoader", '18s'), end='...')

	train_data = TensorDataset(train_seq, train_mask, train_y)											# wrap tensors
	train_sampler = RandomSampler(train_data)															# sampler for sampling the data during training
	train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)				# dataLoader for train set

	val_data = TensorDataset(val_seq, val_mask, val_y)
	val_sampler = SequentialSampler(val_data)
	val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)					# dataLoader for validation set

	test_data = TensorDataset(test_seq, test_mask, test_y)
	test_sampler = SequentialSampler(test_data)
	test_dataloader = DataLoader(test_data, sampler = test_sampler, batch_size=batch_size)				# dataLoader for test set
	
	print(" Elapsed time: {:.3f}".format(timer()-start))
	return train_dataloader, val_dataloader, test_dataloader

def load_pretrained():
	start = timer()
	print(format("Load model", '18s'), end='...\n')

	# import BERT-base pretrained model
	bert = AutoModel.from_pretrained('bert-base-uncased')

	# freeze all the parameters
	for param in bert.parameters():
		param.requires_grad = False

	# Load the BERT tokenizer
	tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

	print("Elapsed time: {:.3f}".format(timer()-start))
	return bert, tokenizer

class BERT_Arch(nn.Module):
	def __init__(self, bert):
		super(BERT_Arch, self).__init__()

		self.bert = bert 
		# dropout layer
		self.dropout = nn.Dropout(0.1)
		# relu activation function
		self.relu =  nn.ReLU()
		# dense layer 1
		self.fc1 = nn.Linear(768,512)
		# dense layer 2 (Output layer)
		self.fc2 = nn.Linear(512,2)
		#softmax activation function
		self.softmax = nn.LogSoftmax(dim=1)

	#define the forward pass
	def forward(self, sent_id, mask):
		#pass the inputs to the model  
		_, cls_hs = self.bert(sent_id, attention_mask=mask)

		x = self.fc1(cls_hs)
		x = self.relu(x)
		x = self.dropout(x)
		# output layer
		x = self.fc2(x)
		# apply softmax activation
		x = self.softmax(x)

		return x

# function to train the model
def train(device, model, optimizer, cross_entropy, train_dataloader):
	print("\nTraining...")
	model.train()

	total_loss, total_accuracy = 0, 0
	# empty list to save model predictions
	total_preds=[]

	start = timer()
	pre_time = start

	# iterate over batches
	for step, batch in enumerate(train_dataloader):
		# progress update after every 50 batches.
		if step % 50 == 0 and not step == 0:
			print('  Batch {:>5,}  of  {:>5,}.  Timer: {:.5f}'.format(step, len(train_dataloader), timer()-pre_time))
			pre_time = timer()

		# push the batch to gpu
		batch = [r.to(device) for r in batch]

		sent_id, mask, labels = batch

		# clear previously calculated gradients 
		model.zero_grad()        

		# get model predictions for the current batch
		preds = model(sent_id, mask)

		# compute the loss between actual and predicted values
		loss = cross_entropy(preds, labels)

		# add on to the total loss
		total_loss = total_loss + loss.item()

		# backward pass to calculate the gradients
		loss.backward()

		# clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
		nn.utils.clip_grad_norm_(model.parameters(), 1.0)

		# update parameters
		optimizer.step()

		# model predictions are stored on GPU. So, push it to CPU
		preds=preds.detach().cpu().numpy()

		# append the model predictions
		total_preds.append(preds)

	# compute the training loss of the epoch
	avg_loss = total_loss / len(train_dataloader)

	# predictions are in the form of (no. of batches, size of batch, no. of classes).
	# reshape the predictions in form of (number of samples, no. of classes)
	total_preds  = np.concatenate(total_preds, axis=0)

	print("Train time: {:.5f}".format(timer()-start))
	#returns the loss and predictions
	return avg_loss, total_preds

# function for evaluating the model
def evaluate(device, model, cross_entropy, val_dataloader):
	print("\nEvaluating...")

	# deactivate dropout layers
	model.eval()

	total_loss, total_accuracy = 0, 0
	# empty list to save the model predictions
	total_preds = []

	start = timer()
	pre_time = start

	# iterate over batches
	for step,batch in enumerate(val_dataloader):
		# Progress update every 50 batches.
		if step % 50 == 0 and not step == 0:
			# Report progress.
			print('  Batch {:>5,}  of  {:>5,}.  Time: {:.5f}'.format(step, len(val_dataloader), timer()-pre_time))
			pre_time = timer()

		# push the batch to gpu
		batch = [t.to(device) for t in batch]

		sent_id, mask, labels = batch

		# deactivate autograd
		with torch.no_grad():
			# model predictions
			preds = model(sent_id, mask)

			# compute the validation loss between actual and predicted values
			loss = cross_entropy(preds,labels)

			total_loss = total_loss + loss.item()

			preds = preds.detach().cpu().numpy()

			total_preds.append(preds)

	# compute the validation loss of the epoch
	avg_loss = total_loss / len(val_dataloader) 

	# reshape the predictions in form of (number of samples, no. of classes)
	total_preds  = np.concatenate(total_preds, axis=0)

	print("Evaluate time: {:.5f}".format(timer()-start))
	return avg_loss, total_preds

def data_processing(df, MAX_LEN, learning_rate, batch_size):
	# load pretrained model
	bert, tokenizer = load_pretrained()

	# split data to train, validate, test
	train_text, train_labels, val_text, val_labels, test_text, test_labels = split_data(df['body'], df['tag'])

	# tokenize data
	tokens_train, tokens_val, tokens_test = tokenize(tokenizer, train_text, val_text, test_text, MAX_LEN)

	# create dataloader
	train_dataloader, val_dataloader, test_dataloader = data_loader(tokens_train, train_labels, tokens_val, val_labels, tokens_test, test_labels, batch_size)

	# pass the pretrained BERT to our define architecture
	model = BERT_Arch(bert)

	# define the optimizer
	optimizer = AdamW(model.parameters(), lr = learning_rate)

	# compute class weight
	class_wts = compute_class_weight('balanced', np.unique(train_labels), train_labels)

	return model, optimizer, class_wts, train_dataloader, val_dataloader, test_dataloader
