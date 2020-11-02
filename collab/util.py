import string
import nltk
import re
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
from timeit import default_timer as timer
from constants import *

from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def clean_content(s):
	"""Given a sentence remove its punctuation and stop words"""
	if not isinstance(s,str):
	  s = str(s)																				# Convert to string
	s = s.lower()																				# Convert to lowercase
	s = s.translate(str.maketrans('','',string.punctuation))									# Remove punctuation
	s = re.sub(r'([\;\:\|â€¢Â«\n])', ' ', s)														# Remove special characters
	s = re.sub(r'(@.*?)[\s]', ' ', s)															# Remove '@name'
	s = re.sub(r'&amp;', '&', s)																# Replace '&amp;' with '&'
	
	tokens = word_tokenize(s)
	stop_words = stopwords.words('english')
	cleaned_s = ' '.join([w for w in tokens if w not in stop_words or w in ['not', 'can']])		# Remove stop-words
	cleaned_s = re.sub(r'\s+', ' ', cleaned_s).strip()											# Replace multi whitespace with single whitespace
	return cleaned_s

def show_preds(preds, test_labels):
	print(f"Accuray: {round(accuracy_score(test_labels, preds), 5) * 100}%")
	print(f"ROC-AUC: {round(roc_auc_score(test_labels, preds), 5) * 100}%")

	fig = plt.figure(figsize=(10,4))
	heatmap = sns.heatmap(data = pd.DataFrame(confusion_matrix(test_labels, preds)), annot = True, fmt = "d", cmap=sns.color_palette("Reds", 50))
	heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha="right", fontsize=14)
	heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha="right", fontsize=14)
	plt.ylabel("Ground Truth")
	plt.xlabel("Prediction")
	plt.show()

# function to predict the model
def predict_model(device, model, test_dataloader):
	total_preds = []
	print("\nPredicting...")
	start = timer()
	pre_time = start
	# iterate over batches
	for step,batch in enumerate(test_dataloader):

		# Progress update every 50 batches.
		if step % 50 == 0 and not step == 0:

			# Report progress.
			print(f"  Batch {step:>5,}  of  {len(test_dataloader):>5,}.  Time: {timer()-pre_time:.3f}")
			pre_time = timer()

		# push the batch to gpu
		batch = [t.to(device) for t in batch]

		sent_id, mask, labels = batch
		with torch.no_grad():
			preds = model(sent_id, mask)
			preds = preds.detach().cpu().numpy()
			total_preds.extend(preds)
	print(f"Predict time: {timer()-start:.3f}")

	predictions = np.argmax(total_preds, axis = 1)

	start = timer()
	print(format("Save Predictions", '18s'), end='...')

	np.save(data_path/"saved_pred", predictions)

	print(f" Elapsed time: {timer()-start:.3f}")

	return predictions

# function to train the model
def train_model(device, model, optimizer, cross_entropy, train_dataloader):
	print("\nTraining...")
	model.train()

	total_loss, total_accuracy = 0, 0
	# empty list to save model predictions
	total_preds=[]
	total_len = len(train_dataloader)

	start = timer()
	pre_time = start

	# iterate over batches
	for step, batch in enumerate(train_dataloader):
		# progress update after every 50 batches.
		if step % 50 == 0 and not step == 0:
			print(f"  Batch {step:>5,}  of  {total_len:>5,}.  Timer: {timer()-pre_time:.3f}")
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

	print(f"  Batch {total_len:>5,}  of  {total_len:>5,}.  Timer: {timer()-pre_time:.3f}")

	# compute the training loss of the epoch
	avg_loss = total_loss / total_len

	# predictions are in the form of (no. of batches, size of batch, no. of classes).
	# reshape the predictions in form of (number of samples, no. of classes)
	total_preds  = np.concatenate(total_preds, axis=0)

	print(f"Train time: {timer()-start:.3f}")
	#returns the loss and predictions
	return avg_loss, total_preds

# function for evaluating the model
def evaluate_model(device, model, cross_entropy, val_dataloader):
	print("\nEvaluating...")

	# deactivate dropout layers
	model.eval()

	total_loss, total_accuracy = 0, 0
	# empty list to save the model predictions
	total_preds = []
	total_len = len(val_dataloader)

	start = timer()
	pre_time = start

	# iterate over batches
	for step,batch in enumerate(val_dataloader):
		# Progress update every 50 batches.
		if step % 50 == 0 and not step == 0:
			# Report progress.
			print(f"  Batch {step:>5,}  of  {total_len:>5,}.  Timer: {timer()-pre_time:.3f}")
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

	print(f"  Batch {total_len:>5,}  of  {total_len:>5,}.  Timer: {timer()-pre_time:.3f}")

	# compute the validation loss of the epoch
	avg_loss = total_loss / total_len

	# reshape the predictions in form of (number of samples, no. of classes)
	total_preds  = np.concatenate(total_preds, axis=0)

	print(f"Evaluate time: {timer()-start:.3f}")
	return avg_loss, total_preds

def train(device, model, optimizer, cross_entropy, epochs, train_dataloader, val_dataloader):
	best_valid_loss = float('inf')
	# empty lists to store training and validation loss of each epoch
	train_losses=[]
	valid_losses=[]

	#for each epoch
	for epoch in range(epochs):

		print(f"\n Epoch {epoch+1} / {epochs}")
		start = timer()
		#train model
		train_loss, _ = train_model(device, model, optimizer, cross_entropy, train_dataloader)

		#evaluate model
		valid_loss, _ = evaluate_model(device, model, cross_entropy, val_dataloader)

		#save the best model
		if valid_loss < best_valid_loss:
			best_valid_loss = valid_loss
		torch.save(model.state_dict(), data_path/weight_file)

		# append training and validation loss
		train_losses.append(train_loss)
		valid_losses.append(valid_loss)

		print(f"\nTraining Loss: {train_loss:.3f}")
		print(f"Validation Loss: {valid_loss:.3f}")
		print(f"Epoch time: {timer()-start:.3f}")
