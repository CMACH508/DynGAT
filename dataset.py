import math
import pandas as pd
import numpy as np
import dgl
import os
from dgl.data import DGLDataset
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
import torch
from utils import date_diff, check_csv, aggregate_csv
from sklearn.model_selection import train_test_split


class Bitcoin2LoopDataset(DGLDataset):
    def __init__(self, raw_dir_, save_dir_):
        super(Bitcoin2LoopDataset, self).__init__(name='bitcoin-2loop_Tmin', url=None,
                                              raw_dir=raw_dir_, save_dir=save_dir_, hash_key=(), force_reload=False,
                                              verbose=False, transform=None)
        self.graph = 0
        self.train_mask = 0
        self.test_mask = 0

    def has_cache(self):
        if os.path.exists(os.path.join(self.save_dir, self.name)):
            print('local cache found, skip processing.')
            return True
        else:
            print('local cache not exists, start processing.')
            return False

    def save(self):
        dgl.save_graphs(os.path.join(self.save_dir, self.name), self.graph)
        print('dgl graph saved to {}'.format(os.path.join(self.save_dir, self.name)))

    def load(self):
        self.train_mask = np.load(os.path.join(self.save_dir, self.name+'_train_mask.npy'), allow_pickle=True)
        self.test_mask = np.load(os.path.join(self.save_dir, self.name+'_test_mask.npy'), allow_pickle=True)
        self.graph = dgl.load_graphs(os.path.join(self.save_dir, self.name))[0][0]
        print('load completed.')

    def process(self):
        if self.has_cache():
            self.load()
            return
        illegal_node = check_csv(os.listdir(os.path.join(self.raw_dir, '钓鱼一阶节点')))
        legal_node = check_csv(os.listdir(os.path.join(self.raw_dir, '非钓鱼一阶节点')))
        all_target_nodes = list(set(illegal_node).union(set(legal_node)))
        #all_transaction = None
        _1loop_illegal = os.path.join(self.raw_dir, '钓鱼一阶节点')
        _1loop_legal = os.path.join(self.raw_dir, '非钓鱼一阶节点')
        all_trans = None
        if os.path.exists(os.path.join(self.raw_dir, 'check_point', '钓鱼一阶节点.csv')):
            illegal_1loop_transaction = pd.read_csv(os.path.join(self.raw_dir, 'check_point', '钓鱼一阶节点.csv'))
            print('钓鱼一阶节点 check point loaded.')
        else:
            illegal_1loop_transaction = aggregate_csv(_1loop_illegal)
            illegal_1loop_transaction.to_csv(os.path.join(self.raw_dir, 'check_point', '钓鱼一阶节点.csv'), index=False)
            print('钓鱼一阶节点 check point saved.')
        all_transaction = illegal_1loop_transaction

        if os.path.exists(os.path.join(self.raw_dir, 'check_point', '非钓鱼一阶节点.csv')):
            legal_1loop_transaction = pd.read_csv(os.path.join(self.raw_dir, 'check_point', '非钓鱼一阶节点.csv'))
            print('非钓鱼一阶节点 check point loaded.')
        else:
            legal_1loop_transaction = aggregate_csv(_1loop_legal)
            legal_1loop_transaction.to_csv(os.path.join(self.raw_dir, 'check_point', '非钓鱼一阶节点.csv'), index=False)
            print('非钓鱼一阶节点 check point saved.')
        all_transaction = pd.concat([all_transaction, legal_1loop_transaction], axis=0)

        if os.path.exists(os.path.join(self.raw_dir, 'check_point', '钓鱼二阶节点.csv')):
            illegal_2loop_transaction = pd.read_csv(os.path.join(self.raw_dir, 'check_point', '钓鱼二阶节点.csv'))
            print('钓鱼二阶节点 check point loaded.')
        else:
            illegal_2loop_transaction = None
            for node in tqdm(illegal_node):
                if os.path.exists(os.path.join(self.raw_dir, '钓鱼二阶节点', node)):
                    df = aggregate_csv(os.path.join(self.raw_dir, '钓鱼二阶节点', node))
                    if illegal_2loop_transaction is not None:
                        illegal_2loop_transaction = pd.concat([illegal_2loop_transaction, df], axis=0)
                    else:
                        illegal_2loop_transaction = df
            illegal_2loop_transaction.to_csv(os.path.join(self.raw_dir, 'check_point', '钓鱼二阶节点.csv'), index=False)
            print('钓鱼二阶节点 check point saved.')
        all_transaction = pd.concat([all_transaction, illegal_2loop_transaction], axis=0)

        if os.path.exists(os.path.join(self.raw_dir, 'check_point', '非钓鱼二阶节点.csv')):
            legal_2loop_transaction = pd.read_csv(os.path.join(self.raw_dir, 'check_point', '非钓鱼二阶节点.csv'))
            print('非钓鱼二阶节点 check point loaded.')
        else:
            legal_2loop_transaction = None
            for node in tqdm(legal_node):
                if os.path.exists(os.path.join(self.raw_dir, '非钓鱼二阶节点', node)):
                    df = aggregate_csv(os.path.join(self.raw_dir, '非钓鱼二阶节点', node))
                    if legal_2loop_transaction is not None:
                        legal_2loop_transaction = pd.concat([legal_2loop_transaction, df], axis=0)
                    else:
                        legal_2loop_transaction = df
            legal_2loop_transaction.to_csv(os.path.join(self.raw_dir, 'check_point', '非钓鱼二阶节点.csv'), index=False)
            print('非钓鱼二阶节点 check point saved.')
        all_transaction = pd.concat([all_transaction, legal_2loop_transaction], axis=0)


        print('trans process complete.')
        print(all_transaction.columns)

        all_nodes = list(set(all_transaction.loc[:, 'From'].values).union(
            set(all_transaction.loc[:, 'To'].values))
        )
        address2nid = {}
        for _, address in tqdm(enumerate(all_nodes)):
            address2nid[address] = _
        all_transaction['From_nid'] = all_transaction['From'].apply(lambda x: address2nid[x])
        all_transaction['To_nid'] = all_transaction['To'].apply(lambda x: address2nid[x])
        src_nid = torch.tensor(all_transaction['From_nid'].values)
        dst_nid = torch.tensor(all_transaction['To_nid'].values)
        self.graph = dgl.graph((src_nid, dst_nid))
        self.graph.edata['feats'] = torch.tensor(all_transaction.loc[:, 'Value'].values).unsqueeze(-1)
        # reset T according to own start time


        #
        T_min = np.min(all_transaction.loc[:, 'TimeStamp'].values)
        print('this is T_max: {}'.format(np.max(all_transaction.loc[:, 'TimeStamp'].values)))
        new_T = torch.tensor(all_transaction.loc[:, 'TimeStamp'].apply(lambda x: x-T_min).values)
        self.graph.edata['T'] = new_T
        self.graph.ndata['feats'] = torch.zeros([self.graph.num_nodes(), 1])
        # edata = height/amount              T = timestamp
        illegal_nid = list(map(lambda x: address2nid[x], illegal_node))
        legal_nid = list(map(lambda x: address2nid[x], legal_node))
        target_nid = list(map(lambda x: address2nid[x], all_target_nodes))
        print(len(illegal_nid), len(legal_nid), len(target_nid))
        self.train_mask, self.test_mask = train_test_split(target_nid, test_size=0.1)
        np.save(os.path.join(self.save_dir, self.name+'_train_mask'), self.train_mask)
        np.save(os.path.join(self.save_dir, self.name+'_test_mask'), self.test_mask)
        label = torch.zeros([self.graph.num_nodes(), 2])
        for i in illegal_nid:
            label[i, 0] = 1
        for i in legal_nid:
            label[i, 1] = 1
        self.graph.ndata['label'] = label.float()


        print('process complete.')

    def __getitem__(self, item):
        assert self.graph != 0, 'graph not exists. Call process first.'
        return self.graph.ndata['feat'][item]

    def __len__(self):
        assert self.graph != 0, 'graph not exists. Call process first.'
        return self.graph.num_nodes()


class EllipticDataset(DGLDataset):
    def __init__(self, raw_dir_, save_dir_):
        super(EllipticDataset, self).__init__(name='bitcoin-elliptic', url=None,
                                              raw_dir=raw_dir_, save_dir=save_dir_, hash_key=(), force_reload=False,
                                              verbose=False, transform=None)
        self.graph = 0
        self.train_mask = 0
        self.test_mask = 0

    def has_cache(self):
        if os.path.exists(os.path.join(self.save_dir, self.name)):
            print('local cache found, skip processing.')
            return True
        else:
            print('local cache not exists, start processing.')
            return False

    def save(self):
        dgl.save_graphs(os.path.join(self.save_dir, self.name), self.graph)
        print('dgl graph saved to {}'.format(os.path.join(self.save_dir, self.name)))

    def load(self):
        self.train_mask = np.load(os.path.join(self.save_dir, self.name+'_train_mask.npy'), allow_pickle=True)
        self.test_mask = np.load(os.path.join(self.save_dir, self.name+'_test_mask.npy'), allow_pickle=True)
        self.graph = dgl.load_graphs(os.path.join(self.save_dir, self.name))[0][0]
        print('load completed.')

    def process(self):
        if self.has_cache():
            self.load()
            return
        classes = pd.read_csv(os.path.join(self.raw_dir, 'elliptic_txs_classes.csv'))
        edge_lists = pd.read_csv(os.path.join(self.raw_dir, 'elliptic_txs_edgelist.csv'))
        features = pd.read_csv(os.path.join(self.raw_dir, 'elliptic_txs_features.csv'), header=None)
        # build dgl graph
        txid2nid = {}
        for _, txid in enumerate(tqdm(features.iloc[:, 0])):
            txid2nid[txid] = _
        src_nid = edge_lists.iloc[:, 0].apply(lambda x: txid2nid[x])
        dst_nid = edge_lists.iloc[:, 1].apply(lambda x: txid2nid[x])
        self.graph = dgl.graph((src_nid, dst_nid))
        # process class
        encoder = OneHotEncoder()
        temp_class = classes.values[:, -1].reshape(-1, 1)
        encoder.fit(temp_class)
        label = encoder.transform(temp_class).toarray()
        label[:, 1] += label[:, -1]
        label = label[:, :-1]
        self.graph.ndata['label'] = torch.tensor(label)
        # process node features
        temp_feat1 = features.values[:, 1].reshape(-1, 1)
        temp_feats = torch.tensor(features.values[:, 2:])
        encoder.fit(temp_feat1)
        n_feats = torch.tensor(encoder.transform(temp_feat1).toarray())
        n_feats = torch.concat([n_feats, temp_feats], dim=1)
        self.graph.ndata['feats'] = n_feats
        self.graph.edata['feats'] = torch.zeros([self.graph.num_edges(), 1])
        self.graph.ndata['T'] = torch.tensor(features.values[:, 1])
        self.train_mask, self.test_mask = train_test_split(np.arange(self.graph.num_nodes()), test_size=0.1)
        np.save(os.path.join(self.save_dir, self.name+'_train_mask'), self.train_mask)
        np.save(os.path.join(self.save_dir, self.name+'_test_mask'), self.test_mask)
        print('process complete.')

    def __getitem__(self, item):
        assert self.graph != 0, 'graph not exists. Call process first.'
        return self.graph.ndata['feat'][item]

    def __len__(self):
        assert self.graph != 0, 'graph not exists. Call process first.'
        return self.graph.num_nodes()


class OTCDataset(DGLDataset):
    def __init__(self, raw_dir_, save_dir_):
        super(OTCDataset, self).__init__(name='bitcoin-otc', url=None,
                                         raw_dir=raw_dir_, save_dir=save_dir_, hash_key=(), force_reload=False,
                                         verbose=False, transform=None)
        self.graph = 0

    def has_cache(self):
        if os.path.exists(os.path.join(self.save_dir, self.name)):
            print('local cache found, skip processing.')
            return True
        else:
            print('local cache not exists, start processing.')
            return False

    def save(self):
        dgl.save_graphs(os.path.join(self.save_dir, self.name), self.graph)
        print('dgl graph saved to {}'.format(os.path.join(self.save_dir, 'otc')))

    def load(self):
        self.graph = dgl.load_graphs(os.path.join(self.save_dir, self.name))[0][0]

    def process(self):
        otc_data = pd.read_csv(os.path.join(self.raw_dir, 'soc-sign-bitcoinotc.csv'), header=None)
        all_txid = np.unique(otc_data.values[:, 0:2].flatten())
        txid2nid = {}
        for _, txid in enumerate(tqdm(all_txid)):
            txid2nid[txid] = _
        src_nid = otc_data.iloc[:, 0].apply(lambda x: txid2nid[x])
        dst_nid = otc_data.iloc[:, 1].apply(lambda x: txid2nid[x])
        self.graph = dgl.graph((src_nid, dst_nid))
        self.graph.ndata['feats'] = torch.zeros(all_txid.shape[0])
        self.graph.edata['rating'] = torch.tensor(otc_data.values[:, 2])
        self.graph.edata['time'] = torch.tensor(otc_data.values[:, -1])
        self.graph.edata['label'] = torch.tensor(otc_data.iloc[0:, 2].apply(lambda x: 1.0 if x > 0 else 0.0))
        print('process complete.')

    def __getitem__(self, item):
        assert self.graph != 0, 'graph not exists. Call process first.'
        return self.graph.edata['rating'][item]

    def __len__(self):
        assert self.graph != 0, 'graph not unexists. Call process first.'
        return self.graph.num_edges()


class AMLSimDataset(DGLDataset):
    def __init__(self, raw_dir_, save_dir_, name_):
        super(AMLSimDataset, self).__init__(name='{}'.format(name_), url=None,
                                         raw_dir=raw_dir_, save_dir=save_dir_, hash_key=(), force_reload=False,
                                         verbose=False, transform=None)
        self.graph = 0

    def has_cache(self):
        if os.path.exists(os.path.join(self.save_dir, self.name)):
            print('local cache found, skip processing.')
            return True
        else:
            print('local cache not exists, start processing.')
            return False

    def save(self):
        dgl.save_graphs(os.path.join(self.save_dir, self.name), self.graph)
        print('dgl graph saved to {}'.format(os.path.join(self.save_dir, self.name)))

    def load(self):
        self.graph = dgl.load_graphs(os.path.join(self.save_dir, self.name))[0][0]

    def process(self):
        transactions = pd.read_csv(os.path.join(self.raw_dir, self.name, 'transactions.csv'))
        accounts = pd.read_csv(os.path.join(self.raw_dir, self.name, 'accounts.csv'))
        sar_accounts = pd.read_csv(os.path.join(self.raw_dir, self.name, 'sar_accounts.csv'))
        alert_transactions = pd.read_csv(os.path.join(self.raw_dir, self.name, 'alert_transactions.csv'))
        alert_accounts = pd.read_csv(os.path.join(self.raw_dir, self.name, 'alert_accounts.csv'))
        src_nid = torch.tensor(transactions.iloc[:, 1].values)
        dst_nid = torch.tensor(transactions.iloc[:, 2].values)
        #if self.name == '10K':
        #    src_nid = torch.concat([src_nid, len(accounts)-1])
        #    dst_nid = torch.concat([src_nid, len(accounts)-1])
        print(torch.max(src_nid))
        print(torch.max(dst_nid))
        self.graph = dgl.graph((src_nid, dst_nid), num_nodes=len(accounts))
        # timestamp -> hour of the day || date
        # edata (base_amt, tran_timestamp)
        # ndata (state, gender, prior_sar_count) (age(birth_date), deposit)
        ndata = torch.tensor(accounts.loc[:, 'initial_deposit'].values).unsqueeze(-1)
        end_date = pd.to_datetime(transactions.loc[:, 'tran_timestamp'].apply(lambda x: x.split('T')[0]).values[-1])
        age = torch.tensor(date_diff(accounts.loc[:, 'birth_date'], end_date, 'Y').astype(int)).unsqueeze(-1)
        ndata = torch.concat([ndata, age], dim=1)

        encoder = OneHotEncoder()
        for feature in ['state', 'gender', 'prior_sar_count']:
            temp = accounts.loc[:, feature].values.reshape(-1, 1)
            encoder.fit(temp)
            temp_tensor = torch.tensor(encoder.transform(temp).toarray())
            ndata = torch.concat([ndata, temp_tensor], dim=1)
        print(ndata.shape, self.graph.num_nodes())
        self.graph.ndata['feats'] = ndata

        trans_hour = transactions.loc[:, 'tran_timestamp'].apply(lambda x: x.split('T')[-1][:-1].split(':')[0]).values
        edata = torch.tensor(trans_hour.astype(int)).unsqueeze(-1)
        amt = torch.tensor(transactions.loc[:, 'base_amt'].values).unsqueeze(-1)
        print(amt.shape)
        edata = torch.concat([edata, amt], dim=1)
        start_date = transactions.loc[:, 'tran_timestamp'].apply(lambda x: x.split('T')[0])
        end_date = start_date.values[-1]
        date = torch.tensor(date_diff(start_date, end_date, 'D')).unsqueeze(-1)

        edata = torch.concat([edata, date], dim=1)
        self.graph.edata['feats'] = edata
        sar_nids = sar_accounts.loc[:, 'ACCOUNT_ID'].values
        label = np.zeros(self.graph.num_nodes())
        for _ in sar_nids:
            label[_] = 1
        label = torch.tensor(label).float()
        self.graph.ndata['label'] = label

        self.train_mask, self.test_mask = train_test_split(np.arange(self.graph.num_nodes()), test_size=0.1)
        np.save(os.path.join(self.save_dir, self.name+'_train_mask'), self.train_mask)
        np.save(os.path.join(self.save_dir, self.name+'_test_mask'), self.test_mask)
        print('AMLSim dataset process complete.')

    def __getitem__(self, item):
        assert self.graph != 0, 'graph not exists. Call process first.'
        return self.graph.ndata['gender'][item]

    def __len__(self):
        assert self.graph != 0, 'graph not unexists. Call process first.'
        return self.graph.num_nodes()


a = 0
b = 0

def T_filter(edges):
    global a
    global b
    if a==0:
        print(edges)
    return (edges.data['T'] >= a) & (edges.data['T'] <= b)

def T_filter_nodes(nodes):
    global a
    global b
    return (nodes.data['T'] >= a) & (nodes.data['T'] <= b)


def cut_graph(graph: dgl.DGLGraph, cuts, T_loc, save_dir=None, add_self_loop=False):
    if T_loc == 'n':
        T_max = torch.max(graph.ndata['T'])
        T_min = torch.min(graph.ndata['T'])
    else:
        T_max = torch.max(graph.edata['T'])
        T_min = torch.min(graph.edata['T'])
    T_n = T_max-T_min+1
    g_list = []
    global a
    global b
    for i in tqdm(range(cuts)):
        a = math.floor(i*T_n/cuts)
        b = math.floor((i*1)*T_n/cuts) - 1
        if T_loc == 'e':
            g_i = dgl.graph(graph.find_edges(graph.filter_edges(T_filter)))
            g_i.ndata['feats'] = graph.ndata['feats']
            g_i.edata = graph.edata[graph.edata['T'] >= a & graph.edata['T'] <= b]
        else:
            g_i = graph.clone()
            n_mask = graph.filter_nodes(T_filter_nodes)
            e_mask = torch.unique(torch.concat([graph.out_edges(n_mask, form='eid'), graph.in_edges(n_mask, form='eid')]))
            g_i.ndata['feats'] = torch.zeros(graph.ndata['feats'].shape)
            g_i.ndata['feats'][n_mask] = graph.ndata['feats'][n_mask].float()
            rm_mask = torch.tensor(list(set(np.arange(graph.num_edges())) - set(e_mask.numpy())), dtype=torch.int64)
            g_i.remove_edges(rm_mask)
        if add_self_loop:
            g_i = dgl.add_self_loop(g_i)
        g_list.append(g_i)
    if save_dir is not None:
        dgl.save_graphs(save_dir, g_list)
    return g_list


def match_dataset_by_name(raw_dataset_name):
    if raw_dataset_name.find('amlsim') != -1:
        return [AMLSimDataset('./datasets/raw/amlsim/', './datasets/processed', '1K'), 'e']
    elif raw_dataset_name.find('elliptic') != -1:
        return [EllipticDataset('./datasets/raw/elliptic_bitcoin_dataset', './datasets/processed/'), 'n']
    else:
        return None


def dataset_exists(name, cuts=None):
    if cuts:
        g_list_save_dir = './datasets/processed/{}-{}'.format(name, cuts)
    else:
        g_list_save_dir = './datasets/processed/{}'.format(name)
    #g_list_save_dir = '../datasets/processed/bitcoin-elliptic-{}-edge_cut'.format(cuts)
    if os.path.exists(g_list_save_dir):
        return True
    else:
        return False

def prepare_dataset(dataset_name, cuts = None):
    dataset, time_pos = match_dataset_by_name(dataset_name)
    dataset.load()
    print(dataset, time_pos)
    g_list_save_dir = './datasets/processed/{}-{}'.format(dataset_name, cuts)
    if not dataset_exists(dataset_name, cuts):
        g_list = cut_graph(dataset.graph, cuts, time_pos, add_self_loop=True, save_dir=g_list_save_dir)
    else:
        g_list = dgl.load_graphs(g_list_save_dir)[0]
    label = dataset.graph.ndata['label'].float()
    mask = {'train':dataset.train_mask, 'test':dataset.test_mask}
    return g_list, label, mask


from utils import to_device
if __name__ == '__main__':
    #EllipticDataset('./datasets/raw/elliptic_bitcoin_dataset', './datasets/processed/')
    #OTCDataset('../datasets/raw/bitcoin-otc', '../datasets/processed/')
    #AMLSimDataset('../datasets/raw/amlsim', '../datasets/processed/', '1K')
    #g = dgl.load_graphs(os.path.join('./datasets/processed/', 'bitcoin-elliptic'))[0][0]
    #print(g)
    #g_list = cut_graph(g, 5, 'n', save_dir='./datasets/processed/bitcoin-elliptic-{}'.format(5))

    #print(dataset_exists('bitcoin-elliptic'))
    #print(dataset_exists('bitcoin-elliptic', 5))
    #print(dataset_exists('amlsim-1k'))
    #print(prepare_dataset('bitcoin-elliptic', 5))
    dataset = prepare_dataset('bitcoin-elliptic', 5)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print((dataset[2]))
    print(to_device(dataset, device))
