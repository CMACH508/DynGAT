import os
import logging
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve, auc

import config


class Logger:
	def __init__(self, args):
		self.log_interval = args['log_interval']
		self.log_path = args['log_path']
		self.experiment_name = args['experiment_name']
		self.loss = []
		self.precision = []
		self.recall = []
		self.show_on_console = args['log_on_console']
		#self.diff_matrix = {'tn':[], 'tp':[], 'fn':[], 'fp':[]}
		if not os.path.exists(self.log_path):
			os.mkdir(self.log_path)
		self.log = logging.getLogger(self.experiment_name)
		self.log_config()

	def log_config(self):
		self.log.setLevel(level= logging.INFO)
		handler = logging.FileHandler(os.path.join(self.log_path, "log.txt"))
		handler.setLevel(level= logging.INFO)
		formatter = logging.Formatter('%(message)s')
		handler.setFormatter(formatter)
		self.log.addHandler(handler)
		if self.show_on_console:
			console_handler = logging.StreamHandler()
			console_handler.setLevel(logging.INFO)
			self.log.addHandler(console_handler)

	def log_epoch(self, e, prediction, label, loss=None, task=['train, test']):
		if loss:
			self.loss.append(loss.item())
		ap = average_precision_score(label.detach().cpu().numpy()[:, 0].astype(int).tolist(), prediction.detach().cpu().numpy()[:, 0].tolist())
		p, r, t = precision_recall_curve(label.detach().cpu().numpy()[:, 0].astype(int).tolist(), prediction.detach().cpu().numpy()[:, 0].tolist())
		pr_auc = auc(r, p)
		if task == 'train':
			self.log.info('epoch {}: {} loss: {}, ap: {},  pr_auc: {}'.format(e, task, self.loss[-1], ap, pr_auc))
		else:
			self.log.info('epoch {}: {} ap: {},  pr_auc: {}'.format(e, task, ap, pr_auc))

	def draw_curve(self, name):
		plt.clf()
		p, r, t = precision_recall_curve(self.label, self.pred)
		plt.grid()
		plt.plot(p, r)
		plt.xlabel('precision')
		plt.ylabel('recall')
		np.save(os.path.join(self.log_dir, '_pred'), self.pred)
		np.save(os.path.join(self.log_dir, '_label'), self.label)
		plt.savefig(os.path.join(self.save_dir, 'pr_curve'))
		#plt.savefig(os.path.join(self.save_dir, 'loss_curve'))

import torch
if __name__ == '__main__':
	logger = Logger(config.params)
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	e = 100
	pred = torch.tensor([0.4,0.45]).float().to(device)
	label = torch.tensor([1.0,0.0]).float().to(device)
	loss = torch.nn.functional.binary_cross_entropy(pred, label)
	mask = [0]

	logger.log_epoch(e, pred, label, loss, 'train')
