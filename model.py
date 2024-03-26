import torch
import time
import numpy as np
import dgl
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import EGATConv,GraphConv
import dgl.function as fn
from dgl.nn.functional import edge_softmax


class GCNGRU(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, cuts):
        super(GCNGRU, self).__init__()
        self.g_rnn = nn.GRU(hidden_size, hidden_size)
        self.gnn_list = nn.ModuleList()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        for _ in range(cuts):
            gnns = nn.ModuleList()
            gnns.append(
                GraphConv(input_size, hidden_size, activation=nn.LeakyReLU())
            )
            for __ in range(num_layers - 1):
                gnns.append(
                    GraphConv(hidden_size, hidden_size, activation=nn.LeakyReLU())
                )
            self.gnn_list.append(gnns)


    def forward(self, g_list):
        feature_list = []
        for _, g in enumerate(g_list):
            n_emb = self.gnn_list[_][0](g, g.ndata['feats'].float())
            for __ in range(1, self.num_layers):
                n_emb = self.gnn_list[_][__](g, n_emb)
            feature_list.append(n_emb)
        feature_list = torch.cat(feature_list)
        feature_list = feature_list.reshape(len(g_list), -1, self.hidden_size)
        gru_out, h_n = self.g_rnn(feature_list)
        return gru_out[-1]


class static_GAT(nn.Module):
    def __init__(self, num_layers, in_n_feats, in_e_feats, hidden_n_feats, hidden_e_feats):
        super(static_GAT, self).__init__()
        self.num_layers = num_layers
        self.gat = nn.ModuleList()
        for i in range(num_layers):
            self.gat.append(EGATConv(hidden_n_feats, in_e_feats, hidden_n_feats, hidden_e_feats,1))
        self.n_linear = nn.Linear(in_n_feats, hidden_n_feats)
        self.norm = nn.ModuleList([nn.BatchNorm1d(hidden_n_feats) for _ in range(num_layers)])

    def forward(self, g):
        #print(self.n_linear.weight)
        #print(self.n_linear.weight.shape, g.ndata['feats'].shape, self.n_linear.weight.dtype,g.ndata['feats'].dtype)
        new_n_feats = self.n_linear(g.ndata['feats'])
        for i in range(self.num_layers):
            new_n_feats, new_e_feats = self.gat[i](g, new_n_feats, g.edata['feats'])
            #print(new_n_feats)
            #print(self.norm)
            new_n_feats = new_n_feats.squeeze(1)
            new_n_feats = self.norm[i](new_n_feats)
            new_n_feats = F.leaky_relu(new_n_feats)
            #new_n_feats = new_n_feats.squeeze(1)
        return new_n_feats


class TimeEncoder(nn.Module):
    def __init__(self, expand_dim_, factor=5):
        super(TimeEncoder, self).__init__()
        self.expand_dim = expand_dim_
        self.factor = factor
        self.basis_freq = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, expand_dim_))).float())
        self.phase = nn.Parameter(torch.zeros(expand_dim_).float())

    def forward(self, ts):
        batch_size = ts.size(0)
        seq_len = ts.size(1)
        ts = ts.view(batch_size, seq_len, 1)
        map_ts = ts * self.basis_freq.view(1, 1, -1)
        harmonic = torch.cos(map_ts)
        return harmonic


class GAT_TE(nn.Module):
    def __init__(self, time_cuts, t_expand_dim, num_layers, in_n_feats, in_e_feats, hidden_n_feats, hidden_e_feats, num_heads,
                 attn_dropout):
        super(GAT_TE, self).__init__()
        self.time_encoder = TimeEncoder(t_expand_dim)
        self.num_layers = num_layers
        self.gat = nn.ModuleList()
        for i in range(time_cuts):
            temp = nn.ModuleList()
            for j in range(num_layers):
                temp.append(EGATConv(hidden_n_feats, in_e_feats, hidden_n_feats, hidden_e_feats, 1))
            self.gat.append(temp)
        self.time_cuts = nn.Parameter(torch.arange(time_cuts).unsqueeze(-1).float())
        self.time_cuts.requires_grad = False
        self.fnh = nn.Linear(in_n_feats, hidden_n_feats)
        self.feh = nn.Linear(in_e_feats, hidden_e_feats)
        self.multiheadAttn = nn.MultiheadAttention(hidden_n_feats + t_expand_dim,
                                                   num_heads, attn_dropout, batch_first=True)
        self.ffn = nn.Linear(time_cuts * (hidden_n_feats + t_expand_dim), hidden_n_feats)
        self.attn_t = 0.0
        self.gnn_t = 0.0

    #def reset_parameters(self):

    def forward(self, g_list):
        feature_list = None
        T_feats = self.time_encoder(self.time_cuts)
        s1 = time.time()
        for _, g in enumerate(g_list):
            n_emb = self.fnh(g.ndata['feats'])
            #print(n_emb[:10,:])
            #e_emb = self.feh(g.edata['feats'])
            for j in range(self.num_layers):
                n_emb, e_emb = self.gat[_][j](g, n_emb, g.edata['feats'])
                n_emb = F.leaky_relu(n_emb)
                n_emb = n_emb.squeeze(1)

            T_feat = T_feats[_]
            T_feat = T_feat.repeat(n_emb.shape[0], 1)
            n_emb = torch.concat([n_emb, T_feat], dim=1).unsqueeze(1)
            #n_emb = n_emb.unsqueeze(1)
            if feature_list is None:
                feature_list = n_emb
            else:
                feature_list = torch.concat([feature_list, n_emb], dim=1)
        s2 = time.time()
        #print(feature_list.shape)
        atten_out, atten_out_weight = self.multiheadAttn(feature_list, feature_list, feature_list)
                # n_emb = self.fnh(blocks[0].srcdata['feats'])
                # e_emb = self.feh(blocks[0].edata['feats'])
                # for j in self.num_layers:
                #	n_emb = self.egat((n_emb,blocks[j].dstdata['feats']))
                #	n_emb = F.leak_relu(n_emb)
                # feature_list.append(n_emb)
        #print(self.attn_t)
        s3 = time.time()
        self.attn_t +=(s3-s2)
        self.gnn_t +=(s2-s1)
        return self.ffn(atten_out.reshape(atten_out.shape[0], -1))
    # multiheadAttn(feature_list)
    # return F.softmax(self.mlp())


class GAT_TE_NoEncoder(nn.Module):
    def __init__(self, time_cuts, t_expand_dim, num_layers, in_n_feats, in_e_feats, hidden_n_feats, hidden_e_feats, num_heads,
                 attn_dropout):
        super(GAT_TE_NoEncoder, self).__init__()
        #self.time_encoder = TimeEncoder(t_expand_dim)
        self.num_layers = num_layers
        self.gat = nn.ModuleList()
        self.norm = nn.ModuleList()
        for i in range(time_cuts):
            temp = nn.ModuleList()
            temp_norm = nn.ModuleList()
            for j in range(num_layers):
                temp.append(EGATConv(hidden_n_feats, in_e_feats, hidden_n_feats, hidden_e_feats, 1))
                temp_norm.append(nn.BatchNorm1d(hidden_n_feats))
            self.gat.append(temp)
            self.norm.append(temp_norm)
        self.time_cuts = nn.Parameter(torch.arange(time_cuts).unsqueeze(-1).float())
        self.time_cuts.requires_grad = False
        self.fnh = nn.Linear(in_n_feats, hidden_n_feats)
        self.feh = nn.Linear(in_e_feats, hidden_e_feats)
        self.multiheadAttn = nn.MultiheadAttention( hidden_n_feats,
                                                   num_heads, attn_dropout, batch_first=True)
        self.ffn = nn.Linear(time_cuts * (hidden_n_feats), hidden_n_feats)



    #def reset_parameters(self):

    def forward(self, g_list):
        feature_list = None
        #T_feats = self.time_encoder(self.time_cuts)
        for _, g in enumerate(g_list):
            n_emb = self.fnh(g.ndata['feats'])
            #print(n_emb[:10,:])
            #e_emb = self.feh(g.edata['feats'])
            for j in range(self.num_layers):
                n_emb, e_emb = self.gat[_][j](g, n_emb, g.edata['feats'])
                n_emb = n_emb.squeeze(1)
                print(n_emb.shape)
                n_emb = self.norm[_][j](n_emb)
                n_emb = F.leaky_relu(n_emb)
                ##n_emb = n_emb.squeeze(1)
            #T_feat = T_feats[_]
            #T_feat = T_feat.repeat(n_emb.shape[0], 1)
            n_emb = n_emb.unsqueeze(1)
            if feature_list is None:
                feature_list = n_emb
            else:
                feature_list = torch.concat([feature_list, n_emb], dim=1)

        #print(feature_list.shape)
        atten_out, atten_out_weight = self.multiheadAttn(feature_list, feature_list, feature_list)
                # n_emb = self.fnh(blocks[0].srcdata['feats'])
                # e_emb = self.feh(blocks[0].edata['feats'])
                # for j in self.num_layers:
                #	n_emb = self.egat((n_emb,blocks[j].dstdata['feats']))
                #	n_emb = F.leak_relu(n_emb)
                # feature_list.append(n_emb)
        return self.ffn(atten_out.reshape(atten_out.shape[0], -1))



class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src["z"], edges.dst["z"]], dim=1)
        a = self.attn_fc(z2)
        return {"e": F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {"z": edges.src["z"], "e": edges.data["e"]}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox["e"], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox["z"], dim=1)
        return {"h": h}

    def forward(self, h):
        # equation (1)
        z = self.fc(h)
        self.g.ndata["z"] = z
        # equation (2)
        self.g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop("h")


class MatGRUCell(torch.nn.Module):
    """
    GRU cell for matrix, similar to the official code.
    Please refer to section 3.4 of the paper for the formula.
    """

    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.update = MatGRUGate(in_feats,
                                 out_feats,
                                 torch.nn.Sigmoid())

        self.reset = MatGRUGate(in_feats,
                                out_feats,
                                torch.nn.Sigmoid())

        self.htilda = MatGRUGate(in_feats,
                                 out_feats,
                                 torch.nn.Tanh())

    def forward(self, prev_Q, z_topk=None):
        if z_topk is None:
            z_topk = prev_Q

        update = self.update(z_topk, prev_Q)
        reset = self.reset(z_topk, prev_Q)

        h_cap = reset * prev_Q
        h_cap = self.htilda(z_topk, h_cap)

        new_Q = (1 - update) * prev_Q + update * h_cap

        return new_Q


class MatGRUGate(torch.nn.Module):
    """
    GRU gate for matrix, similar to the official code.
    Please refer to section 3.4 of the paper for the formula.
    """

    def __init__(self, rows, cols, activation):
        super().__init__()
        self.activation = activation
        self.W = nn.Parameter(torch.Tensor(rows, rows))
        self.U = nn.Parameter(torch.Tensor(rows, rows))
        self.bias = nn.Parameter(torch.Tensor(rows, cols))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.U)
        nn.init.zeros_(self.bias)

    def forward(self, x, hidden):
        out = self.activation(self.W.matmul(x) + \
                              self.U.matmul(hidden) + \
                              self.bias)

        return out


class GAT_FIXED_W(nn.Module):
    def __init__(self, in_n_dim, in_e_dim, out_dim, attn_drop):
        super(GAT_FIXED_W, self).__init__()
        # equation (1)
        self.W = nn.Parameter(torch.zeros([in_n_dim, out_dim]).float())
        self.fc = nn.Linear(in_n_dim, out_dim, bias=False)
        # equation (2)
        self.a = nn.Linear(2 * in_n_dim + in_e_dim, 1, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.activation = nn.LeakyReLU()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.W)

    def message_func(self, edges):
        return {'m': self.a(torch.cat([edges.src['feats'], edges.data['feats'], edges.dst['feats']], dim=1))}

    def forward(self, block: dgl.DGLGraph, n_feats, W_=None):
        with block.local_scope():
            if W_ is None:
                self.reset_parameters()
            else:
                self.W = nn.Parameter(W_)
            block.apply_edges(block, self.message_func)
            block.edata["a"] = edge_softmax(block)
            block.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            rst = block.dstdata["ft"].float()
            return self.activation(rst)


class EGAT_LSTM(nn.Module):
    def __init__(self, num_layers, n_feats_, e_feats_, hidden_feats_, atten_drop_):
        super(EGAT_LSTM, self).__init__()
        self.num_layers = num_layers
        self.GAT = nn.ModuleList()
        self.rnn_layers = nn.ModuleList()
        #self.GAT = nn.ModuleList(GAT_FIXED_W(n_feats_, e_feats_, hidden_feats_, attn_drop=atten_drop_) * num_layers)
        #self.rnn_layers = nn.ModuleList(nn.LSTM(input_size=hidden_feats_, hidden_size=hidden_feats_) * num_layers)
        for i in range(num_layers):
            self.rnn_layers.append(nn.LSTM(input_size=hidden_feats_, hidden_size=hidden_feats_))
            self.GAT.append(GAT_FIXED_W(n_feats_, e_feats_, hidden_feats_, attn_drop=atten_drop_))
        self.W_list = [self.GAT[i].W for i in range(len(self.GAT))]
        self.aggregation = MatGRUCell(hidden_feats_, hidden_feats_)

    def forward(self, g_list):
        feature_list = []
        for g in g_list:
            feature_list.append(g.ndata['feats'])

        #for i in range(time_cuts):
        #    for j in range(self.num_layers):
        #        W = self.W_list[j]
        #        W = self.rnn_layers[j](W)
        #        feature_list[j] = self.GAT[i](g_list[i], feature_list[j], j, w=W)

        for i in range(self.num_layers):
            W = self.W_list[i]
            for j,g in enumerate(g_list):
                W, (h_n, c_n) = self.rnn_layers[i](W)
                feature_list[j] = self.GAT[i](g, feature_list[j], W_=W)
        return feature_list[-1]


class MLP_Classifier(nn.Module):
    def __init__(self, in_feats, classes):
        super(MLP_Classifier, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_feats, int(in_feats/2)), nn.LeakyReLU(),
            nn.Linear(int(in_feats/2), int(in_feats/4), nn.LeakyReLU()),
            nn.Linear(int(in_feats/4), classes), nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.mlp(x)