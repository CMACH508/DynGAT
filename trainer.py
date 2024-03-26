import torch
import utils
import logger
import time
import pandas as pd
import numpy as np
import os
from logger import Logger
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

class Trainer():
	def __init__(self, args, model, classifier, loss_weight, dataset):
		self.args = args
		self.epochs = args['epochs']
		self.model = model
		self.classifier = classifier
		self.loss_weight = loss_weight
		self.dataset = dataset
		self.logger = logger.Logger(args)
		self.save_span = args['save_span']
		self.eval_after_epochs = args['eval_after_epochs']
		self.early_stop_thres = args['early_stop_thres']
		self.model_path = args['log_path']
		self.init_optimizers(args)

	def init_optimizers(self, args):
		params = self.model.parameters()
		self.model_opt = torch.optim.Adam(params, lr=args['learning_rate'], weight_decay=args['weight_decay'])
		params = self.classifier.parameters()
		self.classifier_opt = torch.optim.Adam(params, lr=args['learning_rate'], weight_decay=args['weight_decay'])
		self.model_opt.zero_grad()
		self.classifier_opt.zero_grad()

	def optim_step(self, loss):
		#self.tr_step += 1
		loss.backward()
		self.model_opt.step()
		self.classifier_opt.step()
		self.model_opt.zero_grad()
		self.classifier_opt.zero_grad()

	def save_checkpoint(self):
		torch.save(self.model, os.path.join(self.model_path, 'model.pt'))
		torch.save(self.classifier, os.path.join(self.model_path, 'classifier.pt'))

	def load_checkpoint(self):
		if os.path.isfile(os.path.join(self.model_path, 'model.pt')):
			print("=> loading checkpoint.")
			self.model = torch.load(os.path.join(self.model_path, 'model.pt'))
			self.classifier = torch.load(os.path.join(self.model_path, 'classifier.pt'))
			print("=> loaded checkpoint ")
			return 1
		else:
			print("=> no checkpoint found at '{}'".format(self.model_path))
			return 0

	def run_epoch(self, e, task = ['train', 'test']):
		g_list = self.dataset[0]
		label = self.dataset[1]
		mask = self.dataset[2]
		loss_item = 0.0
		if task == 'train':
			self.model.train()
			self.classifier.train()
			node_emb = self.model(g_list)
			prediction = self.classifier(node_emb)
			loss = F.binary_cross_entropy(prediction[mask['train']], label[mask['train']], weight=self.loss_weight)
			eval_epoch = average_precision_score(label[mask['train']].detach().cpu(), prediction[mask['train']].detach().cpu())
			self.logger.log_epoch(e, prediction[mask['train']], label[mask['train']], loss, task='train')
			#loss = self.comp_loss(pred[mask['train']], label[mask['train']])
			loss_item = loss.item()
			self.optim_step(loss)
		else:
			self.model.eval()
			self.classifier.eval()
			node_emb = self.model(g_list)
			prediction = self.classifier(node_emb)
			eval_epoch = average_precision_score(label[mask['test']].detach().cpu(), prediction[mask['test']].detach().cpu())
			self.logger.log_epoch(e, label[mask['test']], prediction[mask['test']], None, task='test')

		print('ap: {}'.format(eval_epoch))
		return loss_item, prediction

	def train(self, start_from_checkpoint = False):
		#self.tr_step = 0
		best_eval_train = 0
		epochs_without_improve = 0

		if start_from_checkpoint:
			self.load_checkpoint()

		for e in range(self.epochs):
			# train
			eval_train, prediction = self.run_epoch(e, task='train')
			#print('### epoch: ' + str(e) + 'train ap: {}'.format(eval_train))
			if eval_train > best_eval_train:
				best_eval_train = eval_train
			else:
				epochs_without_improve += 1
				if epochs_without_improve > self.early_stop_thres:
					print('### epoch: ' + str(e) + ' Early stop.')
					break
			#log
			#self.logger.log_epoch(e, prediction, label, train_mask, loss, task='train')
			# test
			if e > self.eval_after_epochs:
				eval_test, prediction = self.run_epoch(e, task='test')
			#log
				#self.logger.log_epoch(e, prediction, label, train_mask, loss, task = 'test')
				if not self.save_span == -1 and e%self.save_span == 0:
					self.save_checkpoint()
				#print('### epoch {}: train loss: {}, test loss: {}'.format(e, eval_train, eval_test))
			#else:
			print('### epoch {}: train loss: {}'.format(e, eval_train))
			self.save_checkpoint()


	def test(self):
		self.load_checkpoint()
		eval_test, prediction = self.run_epoch(0, task='test')
		print('### test ap: {}'.format(eval_test))





