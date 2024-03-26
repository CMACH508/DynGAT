import torch
import dgl
import argparse
import warnings
import config
from dataset import prepare_dataset, to_device
from model import MLP_Classifier, GAT_TE
import torch.nn.functional as F
from trainer import Trainer

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')


def run(args, model, target_dataset):
    loss_weight = torch.tensor(args['loss_weight']).float()
    classifier = MLP_Classifier(args['hidden_n_size'], 2)
    gpu = args['gpu']
    model = model.to(gpu)
    classifier = classifier.to(gpu)
    dataset = to_device(target_dataset, gpu)
    loss_weight = loss_weight.to(gpu)
    trainer = Trainer(args, model, classifier, loss_weight, dataset)
    if args['task'] == 'train':
        trainer.train()
    else:
        trainer.test()


if __name__ == '__main__':
    args_ = config.params
    target_dataset_ = prepare_dataset(args_['target_dataset'], args_['time_cuts'])
    model_ = GAT_TE(time_cuts=args_['time_cuts'], t_expand_dim=args_['t_dim'], num_layers=args_['num_gnn_layers'],
                    in_n_feats=target_dataset_[0][0].ndata['feats'].shape[1],
                    in_e_feats=target_dataset_[0][0].edata['feats'].shape[1],
                    hidden_n_feats=args_['hidden_n_size'], hidden_e_feats=args_['hidden_e_size'],
                    num_heads=args_['num_heads'], attn_dropout=args_['attn_dropout'])
    run(args_, model_, target_dataset_)
