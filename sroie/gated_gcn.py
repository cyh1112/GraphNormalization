import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LSTM
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import dgl
import numpy as np


from graph_norm import GraphNorm


"""
    ResGatedGCN: Residual Gated Graph ConvNets
    An Experimental Study of Neural Networks for Variable Graphs (Xavier Bresson and Thomas Laurent, ICLR 2018)
    https://arxiv.org/pdf/1711.07553v2.pdf
"""


class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2): #L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [ nn.Linear( input_dim//2**l , input_dim//2**(l+1) , bias=True ) for l in range(L) ]
        list_FC_layers.append(nn.Linear( input_dim//2**L , output_dim , bias=True ))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        
    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y


class GatedGCNLayer(nn.Module):
    """
        Param: []
    """
    def __init__(self, input_dim, output_dim, dropout, graph_norm, batch_norm, residual=False):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.residual = residual
        
        if input_dim != output_dim:
            self.residual = False
        
        self.A = nn.Linear(input_dim, output_dim, bias=True)
        self.B = nn.Linear(input_dim, output_dim, bias=True)
        self.C = nn.Linear(input_dim, output_dim, bias=True)
        self.D = nn.Linear(input_dim, output_dim, bias=True)
        self.E = nn.Linear(input_dim, output_dim, bias=True)
        
        self.bn_node_h = GraphNorm(output_dim)
        self.bn_node_e = GraphNorm(output_dim)

    def message_func(self, edges):
        Bh_j = edges.src['Bh']    
        e_ij = edges.data['Ce'] +  edges.src['Dh'] + edges.dst['Eh'] # e_ij = Ce_ij + Dhi + Ehj
        edges.data['e'] = e_ij
        return {'Bh_j' : Bh_j, 'e_ij' : e_ij}

    def reduce_func(self, nodes):
        Ah_i = nodes.data['Ah']
        Bh_j = nodes.mailbox['Bh_j']
        e = nodes.mailbox['e_ij'] 
        sigma_ij = torch.sigmoid(e) # sigma_ij = sigmoid(e_ij)
        #h = Ah_i + torch.mean( sigma_ij * Bh_j, dim=1 ) # hi = Ahi + mean_j alpha_ij * Bhj 
        h = Ah_i + torch.sum(sigma_ij * Bh_j, dim=1 ) / ( torch.sum( sigma_ij, dim=1 ) + 1e-6 )  # hi = Ahi + sum_j eta_ij/sum_j' eta_ij' * Bhj <= dense attention       
        return {'h' : h}
    
    def forward(self, g, h, e, snorm_n, snorm_e, graph_node_size, graph_edge_size):
        
        h_in = h # for residual connection
        e_in = e # for residual connection
        
        g.ndata['h']  = h 
        g.ndata['Ah'] = self.A(h) 
        g.ndata['Bh'] = self.B(h) 
        g.ndata['Dh'] = self.D(h)
        g.ndata['Eh'] = self.E(h) 
        g.edata['e']  = e 
        g.edata['Ce'] = self.C(e) 
        g.update_all(self.message_func,self.reduce_func) 
        h = g.ndata['h'] # result of graph convolution
        e = g.edata['e'] # result of graph convolution
        
        if self.graph_norm:
            h = h * snorm_n # normalize activation w.r.t. graph size
            e = e * snorm_e # normalize activation w.r.t. graph size
        
        if self.batch_norm:
            h = self.bn_node_h(h, graph_node_size) # graph normalization  
            e = self.bn_node_e(e, graph_edge_size) # graph normalization  

        
        h = F.relu(h) # non-linear activation
        e = F.relu(e) # non-linear activation
        
        if self.residual:
            h = h_in + h # residual connection
            e = e_in + e # residual connection
        
        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)
        
        return h, e
    
    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels)


class DenseLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # self.bn = nn.BatchNorm1d(in_dim)
        self.bn = nn.LayerNorm(in_dim)
        self.linear = nn.Linear(in_dim, out_dim)
    def forward(self, feat):
        feat = self.bn(feat)
        feat = F.relu(feat)
        feat = self.linear(feat)
        return feat

class GatedGCNNet(nn.Module):
    
    def __init__(self, net_params):
        super().__init__()
        in_dim_text = net_params['in_dim_text']
        in_dim_node = net_params['in_dim_node'] # node_dim (feat is an integer)
        in_dim_edge = net_params['in_dim_edge'] # edge_dim (feat is a float)
        hidden_dim = net_params['hidden_dim']
        n_classes = net_params['n_classes']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.ohem = net_params['OHEM']

        self.readout = net_params['readout']
        self.graph_norm = net_params['graph_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.n_classes = n_classes
        self.device = net_params['device']
        
        self.embedding_text = nn.Embedding(in_dim_text, hidden_dim) # node feat is an integer
        self.embedding_h = nn.Linear(in_dim_node, hidden_dim) # edge feat is a float
        self.embedding_e = nn.Linear(in_dim_edge, hidden_dim) # edge feat is a float
        self.layers = nn.ModuleList([ GatedGCNLayer(hidden_dim, hidden_dim, dropout,
                                                       self.graph_norm, self.batch_norm, self.residual) for _ in range(n_layers) ])
        self.dense_layers = nn.ModuleList([
                    DenseLayer(hidden_dim + i * hidden_dim, hidden_dim) for i in range(1, n_layers+1)])
        
        self.lstm = LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.MLP_layer = MLPReadout(hidden_dim, n_classes)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def lstm_text_embeding(self, text, text_length):
        packed_sequence = pack_padded_sequence(text, text_length, batch_first=True, enforce_sorted=False)
        outputs_packed, (h_last, c_last) = self.lstm(packed_sequence)
        # outputs, _ = pad_packed_sequence(outputs_packed)
        return h_last.mean(0)

    def clamp(self):
        min = torch.tensor(0.).cuda()
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, UnifiedNorm):
                    m.lambda_batch.masked_fill_(m.lambda_batch < 0, min)
                    m.lambda_graph.masked_fill_(m.lambda_graph < 0, min)
                    m.lambda_adja.masked_fill_(m.lambda_adja < 0, min)
                    m.lambda_node.masked_fill_(m.lambda_node < 0, min)


    def concat(self, h_list, l):
        h_concat = torch.cat(h_list, dim=1)
        h = self.dense_layers[l](h_concat)
        return h

    def forward(self, g, h, e, text, text_length, snorm_n, snorm_e, graph_node_size, graph_edge_size):
        # input embedding
        h_embeding = self.embedding_h(h)
        e_embeding = self.embedding_e(e)

        text_embeding = self.embedding_text(text)
        text_embeding = self.lstm_text_embeding(text_embeding, text_length)

        text_embeding = F.normalize(text_embeding)

        e = e_embeding
        h = h_embeding + text_embeding
        all_h = [h]
        for i, conv in enumerate(self.layers):
            h1, e = conv(g, h, e, snorm_n, snorm_e, graph_node_size, graph_edge_size)
            all_h.append(h1)
            h = self.concat(all_h, i)

        # output
        h_out = self.MLP_layer(h)

        return h_out
        
    def _ohem(self, pred, label):
        # import pdb; pdb.set_trace()
        pred = pred.data.cpu().numpy()
        label = label.data.cpu().numpy()

        pos_num = sum(label != 0)
        neg_num = pos_num * self.ohem

        pred_value = pred[:, 1:].max(1)

        neg_score_sorted = np.sort(-pred_value[label == 0])

        if neg_score_sorted.shape[0] > neg_num:
            threshold = -neg_score_sorted[neg_num - 1]
            mask = ((pred_value >= threshold) | (label != 0))
        else:
            mask = label != -1
        return torch.from_numpy(mask)

    def loss(self, pred, label):

        mask_label = label.clone()
        mask = self._ohem(pred, label)
        mask = mask.to(pred.device)
        mask_label[mask == False] = -100
        loss = self.criterion(pred, mask_label)

        # calculating label weights for weighted loss computation
        # V = label.size(0)
        # label_count = torch.bincount(label)
        # label_count = label_count[label_count.nonzero()].squeeze()
        # cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        # cluster_sizes[torch.unique(label)] = label_count
        # weight = (V - cluster_sizes).float() / V
        # weight *= (cluster_sizes>0).float()
        
        # # weighted cross-entropy for unbalanced classes
        # criterion = nn.CrossEntropyLoss(weight=weight)
        # loss = criterion(pred, label)

        return loss




if __name__ == '__main__':

    net_params = {}
    net_params['in_dim'] = 1
    net_params['hidden_dim'] = 256
    net_params['out_dim'] = 256
    net_params['n_classes'] = 5
    net_params['in_feat_dropout'] = 0.1
    net_params['dropout'] = 0.1
    net_params['L'] = 5
    net_params['readout'] = True
    net_params['graph_norm'] = True
    net_params['batch_norm'] = True
    net_params['residual'] = True
    net_params['device'] = 'cuda'

    net = GatedGCNNet(net_params)
    print(net)

