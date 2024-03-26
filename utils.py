import os
import pandas as pd
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.metrics import average_precision_score, precision_recall_curve, auc


class Metric:
    def __init__(self, save_dir_):
        self.pred = []
        self.label = []
        self.save_dir = save_dir_

    def append(self, pred_, label_):
        self.pred.extend(pred_)
        self.label.extend(label_)

    def draw_curve(self, name):
        plt.clf()
        p, r, t = precision_recall_curve(self.label, self.pred)
        ap = average_precision_score(self.label, self.pred)
        plt.grid()
        plt.plot(p, r)
        plt.xlabel('precision')
        plt.ylabel('recall')
        #plt.annotate("ap: {}".format(ap))
        plt.savefig(os.path.join(self.save_dir, name))
        # ll = np.array([p,r,t])
        np.save(os.path.join(self.save_dir, 'p'),p)
        np.save(os.path.join(self.save_dir,'r'),r)
        np.save(os.path.join(self.save_dir,'t'),t)
        print('pr auc is {}'.format(auc(r,p)))
        np.save(os.path.join(self.save_dir,'pred'), self.pred)
        np.save(os.path.join(self.save_dir,'label'), self.label)
        return p, r, t


def date_diff(start_date, end_date, gap):
    assert gap in ['D', 'M', 'Y'], 'illegal gap'
    df = pd.DataFrame(start_date.values, columns=['start_date'])
    df['end_date'] = end_date
    df = df.apply(pd.to_datetime)
    df['diff'] = (df['end_date'] - df['start_date']) / np.timedelta64(1, gap)
    return df['diff'].values


#def to_device(g_list, device: torch.device):
#    g_list_cuda = []
#    for g in g_list:
#        g_list_cuda.append(g.to(device))
#        g_list_cuda[-1].ndata['feats'] = g_list_cuda[-1].ndata['feats'].float().to(device)
#        g_list_cuda[-1].edata['feats'] = g_list_cuda[-1].edata['feats'].float().to(device)
#    return g_list_cuda


def to_device(dataset, device: torch.device):
    g_list = dataset[0]
    label = dataset[1]
    mask = dataset[2]
    g_list_cuda = []
    for g in g_list:
        g_list_cuda.append(g.to(device))
        g_list_cuda[-1].ndata['feats'] = g_list_cuda[-1].ndata['feats'].float().to(device)
        g_list_cuda[-1].edata['feats'] = g_list_cuda[-1].edata['feats'].float().to(device)
    #mask_ = {'train':,'test':}
    #print(mask)
    #mask['train'] = torch.tensor(mask['train']).float().to(device)
    #mask['test'] = torch.tensor(mask['test']).float().to(device)
    label = g_list[0].ndata['label'].float().to(device)
    return [g_list_cuda, label, mask]



def check_csv(name_list, to_name=True):
    new_csv_list = []
    for i in name_list:
        if i.split('.')[-1] == 'csv':
            if to_name == True:
                new_csv_list.append(i.split('.')[0])
            else:
                new_csv_list.append(i)
    return new_csv_list


def aggregate_csv(folder_path):
    aggregation = None
    csv_list = check_csv(os.listdir(folder_path), to_name=False)
    for csv in csv_list:
        df = pd.read_csv(os.path.join(folder_path, csv))
        if aggregation is None:
            aggregation = df
        else:
            aggregation = pd.concat([aggregation, df], axis=0)
    return aggregation

