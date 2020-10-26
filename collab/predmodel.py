import torch
from timeit import default_timer as timer

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

def predict(device, model, test_dataloader):
	total_preds = []
	print("\nPredicting...")
	start = timer()
	pre_time = start
	# iterate over batches
	for step,batch in enumerate(test_dataloader):

		# Progress update every 50 batches.
		if step % 50 == 0 and not step == 0:

			# Report progress.
			print('  Batch {:>5,}  of  {:>5,}.  Time: {:.5f}'.format(step, len(test_dataloader), timer()-pre_time))
			pre_time = timer()

		# push the batch to gpu
		batch = [t.to(device) for t in batch]

		sent_id, mask, labels = batch
		with torch.no_grad():
			preds = model(sent_id, mask)
			preds = preds.detach().cpu().numpy()
			total_preds.extend(preds)
	print("Predict time: {:.5f}".format(timer()-start))
	return total_preds

def show_preds(total_preds, test_y):
	preds = np.argmax(total_preds, axis = 1)

	print(f"""Accuray: {round(accuracy_score(test_y, preds), 5) * 100}%
	ROC-AUC: {round(roc_auc_score(test_y, preds), 5) * 100}%""")

	fig = plt.figure(figsize=(10,4))
	heatmap = sns.heatmap(data = pd.DataFrame(confusion_matrix(test_y, preds)), annot = True, fmt = "d", cmap=sns.color_palette("Reds", 50))
	heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
	heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=14)
	plt.ylabel('Ground Truth')
	plt.xlabel('Prediction')
	plt.show()
	