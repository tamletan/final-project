import torch
import preprocess as pproc
import dataprocess as dproc
import torch.nn as nn
from timeit import default_timer as timer
import argparse

if __name__ == '__main__':

	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--data", type=str, default="./dataset.csv", help="dataset path")
	ap.add_argument("-l", "--maxlen", type=int, default=512, help="max seq len")
	ap.add_argument("-b", "--batch", type=int, default=32, help="batch size")
	ap.add_argument("-r", "--rate", type=float, default=2e-5, help="learning rate")
	ap.add_argument("-e", "--epochs", type=int, default=4, help="number of epochs")
	args = vars(ap.parse_args())

	MAX_LEN = args["maxlen"]
	# define a batch size
	batch_size = args["batch"]
	# learning rate
	learning_rate = args["rate"]
	epochs = args["epochs"]

	device = torch.device("cuda")
	print(torch.cuda.get_device_name(0))

	df = pproc.load_data(args["data"])

	print('\n','='*50,'\n')

	model, optimizer, class_wts, train_dataloader, val_dataloader, test_dataloader = dproc.data_processing(df, MAX_LEN, learning_rate, batch_size)

	# push the model to GPU
	model = model.to(device)

	# convert class weight to tensor
	weights= torch.tensor(class_wts,dtype=torch.float)
	weights = weights.to(device)

	# loss function
	cross_entropy  = nn.NLLLoss(weight=weights)

	print('\n','='*50,'\n')

	best_valid_loss = float('inf')

	# empty lists to store training and validation loss of each epoch
	train_losses=[]
	valid_losses=[]

	#for each epoch
	for epoch in range(epochs):
	    
	  print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
	  start = timer()
	  #train model
	  train_loss, _ = dproc.train(device, model, optimizer, cross_entropy, train_dataloader)
	  
	  #evaluate model
	  valid_loss, _ = dproc.evaluate(device, model, cross_entropy, val_dataloader)
	  
	  #save the best model
	  if valid_loss < best_valid_loss:
	    best_valid_loss = valid_loss
	    torch.save(model.state_dict(), './saved_weights.pt')
	  
	  # append training and validation loss
	  train_losses.append(train_loss)
	  valid_losses.append(valid_loss)
	  
	  print(f'\nTraining Loss: {train_loss:.3f}')
	  print(f'Validation Loss: {valid_loss:.3f}')
	  print("Epoch time: {:.5f}".format(timer()-start))