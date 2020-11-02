import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from util import clean_content
from timeit import default_timer as timer
from constants import *

from BERT_Arch import BERT_Arch
from transformers import BertModel, BertTokenizerFast, AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

def load_data(path):
	df = pd.read_csv(path)
	print(f"Number of training rows: {df.shape[0]:,}\n")

	# check class distribution
	print(df["tag"].value_counts(normalize = True))
	df["tag"].replace({"legit": 0, "spam":1}, inplace=True)

	print("\nDrop {:,} row with null value\n".format(df["body"].isnull().sum()))
	df.dropna(subset=["body"], inplace=True)
	print(f"Number of remain rows: {df.shape[0]:,}\n")

	start = timer()
	print(format("Clean data", '18s'), end='...')

	df["body"] = df["body"].apply(clean_content)
	df.replace("", float("NaN"), inplace=True)
	df.dropna(subset = ["body"], inplace=True)

	print(f" Elapsed time: {timer()-start:.3f}")
	print(f"Number of remain rows: {df.shape[0]:,}\n")

	return df

def split_data(body, tag):
	state = 2020
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

	print(f" Elapsed time: {timer()-start:.3f}")

	return train_text, train_labels, val_text, val_labels, test_text, test_labels

def tokens(tokenizer, data, max_len):
	token_ = tokenizer.batch_encode_plus(
		data.tolist(),
		max_length = max_len,
		padding=True,
		truncation=True,
		return_token_type_ids=False
	)
	return token_

def tokenize(train_text, val_text, test_text, max_len):
	print(format("Load Tokenizer", '18s'), end='...\n')
	tokenizer = BertTokenizerFast.from_pretrained(bert_pretrain)

	start = timer()
	print(format("Tokenize", '18s'), end='...')

	# tokenize and encode sequences
	tokens_train = tokens(tokenizer, train_text, max_len)
	tokens_val = tokens(tokenizer, val_text, max_len)
	tokens_test = tokens(tokenizer, test_text, max_len)

	print(f" Elapsed time: {timer()-start:.3f}")

	return tokens_train, tokens_val, tokens_test

def create_loader(tokens, labels, batch_size):
	seq = torch.tensor(tokens['input_ids'])
	mask = torch.tensor(tokens['attention_mask'])
	label_tensor = torch.tensor(labels.tolist())

	data = TensorDataset(seq, mask, label_tensor)								# wrap tensors
	sampler = RandomSampler(data)												# sampler for sampling the data during training
	dataloader = DataLoader(data, sampler = sampler, batch_size = batch_size)

	return dataloader

def data_loader(tokens_train, train_labels, tokens_val, val_labels, tokens_test, test_labels, batch_size):
	start = timer()
	print(format("Create DataLoader", '18s'), end='...')

	train_dataloader = create_loader(tokens_train, train_labels, batch_size)
	val_dataloader = create_loader(tokens_val, val_labels, batch_size)
	test_dataloader = create_loader(tokens_test, test_labels, batch_size)

	print(f" Elapsed time: {timer()-start:.3f}")

	start = timer()
	print(format("Save DataLoader", '18s'), end='...')

	torch.save(train_dataloader, data_path/train_loader_file)
	torch.save(val_dataloader, data_path/val_loader_file)
	torch.save(test_dataloader, data_path/test_loader_file)
	
	test_labels.to_pickle(data_path/test_label_file)
	train_labels.to_pickle(data_path/train_label_file)

	print(f" Elapsed time: {timer()-start:.3f}")

def loss_func(device, train_labels):
	# compute class weight
	class_wts = compute_class_weight('balanced', np.unique(train_labels), train_labels)
	# convert class weight to tensor
	weights= torch.tensor(class_wts, dtype = torch.float)
	weights = weights.to(device)
	# loss function
	cross_entropy  = nn.NLLLoss(weight = weights)
	
	return cross_entropy

def initial_model(device, learning_rate):
    start = timer()
    try:
        model = torch.load(data_path/model_file)
    except:
        bert = BertModel.from_pretrained(bert_pretrain)
        for param in bert.parameters():
            param.requires_grad = False
    
        model = BERT_Arch(bert)
        torch.save(model, data_path/model_file)

    print(f" Elapsed time: {timer()-start:.3f}")
    model = model.to(device)
    
    start = timer()
    # define the optimizer
    optimizer = AdamW(model.parameters(), lr = learning_rate)
    print(f" Elapsed time: {timer()-start:.3f}")
    return model, optimizer
    
def procedure(path, max_len, batch_size):
	df = load_data(path)
	train_text, train_labels, val_text, val_labels, test_text, test_labels = split_data(df['body'], df['tag'])
	tokens_train, tokens_val, tokens_test = tokenize(train_text, val_text, test_text, max_len)
	data_loader(tokens_train, train_labels, tokens_val, val_labels, tokens_test, test_labels, batch_size)
