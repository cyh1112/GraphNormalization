import torch.nn as nn
import torch

# Adjance norm for node
class AdjaNodeNorm(nn.Module):

    def __init__(self, num_features, eps=1e-5, affine=True):
        super(AdjaNodeNorm, self).__init__()
        self.eps = eps
        self.affine = affine
        self.num_features = num_features

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(self.num_features))
            self.beta = nn.Parameter(torch.zeros(self.num_features))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)


    def message_func(self, edges):
        return {"h": edges.src["norm_h"]}

    def reduce_func(self, nodes):
        dst_h = nodes.mailbox['h']
        src_h = nodes.data['h']

        h = torch.cat([dst_h, src_h.unsqueeze(1)], 1)
        mean = torch.mean(h, dim=(1, 2))
        var = torch.std(h, dim=(1, 2))

        mean = mean.unsqueeze(1).expand_as(src_h)
        var = var.unsqueeze(1).expand_as(src_h)
        return {"norm_mean": mean, "norm_var": var}

    def forward(self, g, h):
        g.ndata["norm_h"] = h
        g.update_all(self.message_func, self.reduce_func)

        mean = g.ndata['norm_mean']
        var = g.ndata['norm_var']

        norm_h = (h - mean) / (var + self.eps)

        if self.affine:
            return self.gamma * norm_h + self.beta
        else:
            return norm_h

# Adjance norm for edge
class AdjaEdgeNorm(nn.Module):

    def __init__(self, num_features, eps=1e-5, affine=True):
        super(AdjaEdgeNorm, self).__init__()
        self.eps = eps
        self.affine = affine
        self.num_features = num_features

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(self.num_features))
            self.beta = nn.Parameter(torch.zeros(self.num_features))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

    def message_func(self, edges):
        return {"e": edges.data['norm_e']}

    def reduce_func(self, nodes):
        e = nodes.mailbox['e']
        mean = torch.mean(e, dim=(1, 2))
        var = torch.std(e, dim=(1, 2))

        mean = mean.unsqueeze(1).expand_as(e[:, 0, :])
        var = var.unsqueeze(1).expand_as(e[:, 0, :])
        return {"norm_mean": mean, "norm_var": var}

    def apply_edges(self, edges):
        mean = edges.dst['norm_mean']
        var = edges.dst['norm_var']
        return {"edge_mean": mean, "edge_var": var}

    def forward(self, g, e):
        g.edata['norm_e'] = e
        g.update_all(self.message_func, self.reduce_func)
        g.apply_edges(self.apply_edges)
        mean = g.edata['edge_mean']
        var = g.edata['edge_var']

        norm_e = (e - mean) / (var + self.eps)
        if self.affine:
            return self.gamma * norm_e + self.beta
        else:
            return norm_e
