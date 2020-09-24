import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dgl.nn.pytorch import GATConv


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

class GATHeadLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, graph_norm, batch_norm):
        super().__init__()
        self.dropout = dropout
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        
        self.wq = nn.Linear(in_dim, out_dim, bias=False)
        self.wk = nn.Linear(in_dim, out_dim, bias=False)

        self.attn_fc = nn.Linear(out_dim, 1, bias=False)
        self.batchnorm_h = nn.BatchNorm1d(out_dim)

    def edge_attention(self, edges):
        # import pdb; pdb.set_trace()
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    # def edge_self_attention(self, edges):
    #     # import pdb; pdb.set_trace()
    #     a = torch.sum(edges.src['z'] * edges.dst['z'], 1, keepdim=True)
    #     return {'e': a}

    def edge_self_attention(self, edges):
        import pdb; pdb.set_trace()
        
        a = edges.src['z'] * edges.dst['z']
        # a = self.attn_fc(a)
        return {'e': a}

    def edge_qk_attention(self, edges):
        # import pdb; pdb.set_trace()
        q = self.wq(edges.src['h'])
        k = self.wk(edges.dst['h'])
        a = torch.sum(q * k, 1, keepdim=True)
        return {'e': a}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # import pdb; pdb.set_trace()
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        alpha = F.dropout(alpha, self.dropout, training=self.training)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, h, snorm_n):
        # g.ndata['h'] = h
        z = self.fc(h)
        g.ndata['z'] = z
        g.apply_edges(self.edge_self_attention)
        g.update_all(self.message_func, self.reduce_func)
        h = g.ndata['h']
        if self.graph_norm:
            h = h * snorm_n
        if self.batch_norm:
            h = self.batchnorm_h(h)
        h = F.elu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class GATLayer(nn.Module):
    """
        Param: [in_dim, out_dim, n_heads]
    """
    def __init__(self, in_dim, out_dim, num_heads, dropout, graph_norm, batch_norm, residual=False, activation=None, dgl_builtin=False):

        super().__init__()
        self.dgl_builtin = dgl_builtin

        if dgl_builtin == False:
            self.in_channels = in_dim
            self.out_channels = out_dim
            self.num_heads = num_heads
            self.residual = residual
            
            if in_dim != (out_dim*num_heads):
                self.residual = False
            
            self.heads = nn.ModuleList()
            for i in range(num_heads):
                self.heads.append(GATHeadLayer(in_dim, out_dim, dropout, graph_norm, batch_norm))
            self.merge = 'cat' 

        else:
            self.in_channels = in_dim
            self.out_channels = out_dim
            self.num_heads = num_heads
            self.residual = residual
            self.activation = activation
            self.graph_norm = graph_norm
            self.batch_norm = batch_norm
            
            if in_dim != (out_dim*num_heads):
                self.residual = False

            # Both feat and weighting dropout tied together here
            self.conv = GATConv(in_dim, out_dim, num_heads, dropout, dropout)
            self.batchnorm_h = nn.BatchNorm1d(out_dim)



    def forward(self, g, h, snorm_n):
        if self.dgl_builtin == False:
            h_in = h # for residual connection
            head_outs = [attn_head(g, h, snorm_n) for attn_head in self.heads]
            
            if self.merge == 'cat':
                h = torch.cat(head_outs, dim=1)
            else:
                h = torch.mean(torch.stack(head_outs))
            
            if self.residual:
                h = h_in + h # residual connection
            return h
        else:
            h_in = h # for residual connection

            h = self.conv(g, h).flatten(1)

            if self.graph_norm:
                h = h * snorm_n
            if self.batch_norm:
                h = self.batchnorm_h(h)
            
            if self.residual:
                h = h_in + h # residual connection

            if self.activation:
                h = self.activation(h)
            return h
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads, self.residual)


class GATNet(nn.Module):

    def __init__(self, net_params):
        super().__init__()

        in_dim_node = net_params['in_dim'] # node_dim (feat is an integer)
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        num_heads = net_params['n_heads']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']

        self.readout = net_params['readout']
        self.graph_norm = net_params['graph_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.dropout = dropout
        self.n_classes = n_classes
        self.device = net_params['device']
        
        self.text_embedding = nn.Embedding(in_dim_node, hidden_dim * num_heads) # node feat is an integer
        self.feat_embedding = nn.Linear(10, hidden_dim * num_heads, bias=True)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([GATLayer(hidden_dim * num_heads, hidden_dim, num_heads,
                                              dropout, self.graph_norm, self.batch_norm, self.residual) for _ in range(n_layers-1)])
        self.layers.append(GATLayer(hidden_dim * num_heads, out_dim, 1, dropout, self.graph_norm, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(out_dim, n_classes)

        # self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def _group_mean(self, text_embeding, text_length):
        mean_embedding = []
        count = 0
        for i in range(text_length.shape[0]):
            mean_embedding.append(text_embeding[count:text_length[i]].mean(0).unsqueeze(0))

        return torch.cat(mean_embedding, 0)

    def _group_sum(self, text_embeding, text_length):
        mean_embedding = []
        count = 0
        for i in range(text_length.shape[0]):
            mean_embedding.append(text_embeding[count:text_length[i]].sum(0).unsqueeze(0))

        return torch.cat(mean_embedding, 0)

    def _group_max(self, text_embeding, text_length):
        mean_embedding = []
        count = 0
        for i in range(text_length.shape[0]):
            mean_embedding.append(text_embeding[count:text_length[i]].max(0).unsqueeze(0))

        return torch.cat(mean_embedding, 0)

    #batch_graphs, boxes, text, text_length, batch_snorm_n, batch_snorm_e
    def forward(self, g, x, e, text, text_length, snorm_n, snorm_e):
    
        feat_embeding = self.feat_embedding(x)
        # h = box_embeding

        text_embeding = self.text_embedding(text)
        text_embeding = self._group_mean(text_embeding, text_length)

        h_embeding = text_embeding + feat_embeding

        h = self.in_feat_dropout(h_embeding)

        # GAT
        for gat_layer in self.layers:
            h = gat_layer(g, h, snorm_n)
            if h.shape[-1] == feat_embeding.shape[-1]:
                h = h + feat_embeding
            
        # output
        h_out = self.MLP_layer(h)

        return h_out
    
    def _ohem(self, pred, label):
        # import pdb; pdb.set_trace()
        pred = pred.data.cpu().numpy()
        label = label.data.cpu().numpy()

        pos_num = sum(label != 0)
        neg_num = pos_num * 3

        pred_cls = pred.argmax(1)
        pred_value = pred.max(1)

        neg_score_sorted = np.sort(-pred_value[label == 0])

        if neg_score_sorted.shape[0] > neg_num:
            threshold = -neg_score_sorted[neg_num - 1]
            mask = ((pred_value >= threshold) | (label != 0))
        else:
            mask = label != -1

        return torch.from_numpy(mask)
    
    def loss(self, pred, label):

        # mask_label = label.clone()
        # mask = self._ohem(pred, label)
        # mask = mask.to(pred.device)
        # mask_label[mask == False] = -100
        # loss = self.criterion(pred, mask_label)


        # calculating label weights for weighted loss computation
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes > 0).float()
        
        # weighted cross-entropy for unbalanced classes
        criterion = nn.CrossEntropyLoss(weight=weight)
        loss = criterion(pred, label)
        return loss


if __name__ == '__main__':

    net_params = {}
    net_params['in_dim'] = 256
    net_params['hidden_dim'] = 256
    net_params['out_dim'] = 256
    net_params['n_classes'] = 5
    net_params['n_heads'] = 8
    net_params['in_feat_dropout'] = 0.1
    net_params['dropout'] = 0.1
    net_params['L'] = 5
    net_params['readout'] = True
    net_params['graph_norm'] = True
    net_params['batch_norm'] = True
    net_params['residual'] = True
    net_params['device'] = 'cuda'

    net = GATNet(net_params)

