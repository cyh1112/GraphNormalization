import torch.nn as nn
import torch
from norm.graph_norm import GraphNorm
from norm.adjance_norm import AdjaNodeNorm, AdjaEdgeNorm

class UnitedNormBase(nn.Module):

    def __init__(self, num_features, is_node=True):
        super(UnitedNormBase, self).__init__()
        self.clamp = False
        self.num_features = num_features
        self.gamma = nn.Parameter(torch.ones(self.num_features))
        self.beta = nn.Parameter(torch.zeros(self.num_features))

        self.lambda_batch = nn.Parameter(torch.ones(self.num_features))
        self.lambda_graph = nn.Parameter(torch.ones(self.num_features))
        self.lambda_adja = nn.Parameter(torch.ones(self.num_features))
        self.lambda_node = nn.Parameter(torch.ones(self.num_features))

        self.batch_norm = nn.BatchNorm1d(self.num_features, affine=False)
        self.graph_norm = GraphNorm(self.num_features, is_node=is_node, affine=False)
        self.node_norm = nn.LayerNorm(self.num_features, elementwise_affine=False)
        if is_node:
            self.adja_norm = AdjaNodeNorm(self.num_features, affine=False)
        else:
            self.adja_norm = AdjaEdgeNorm(self.num_features, affine=False)

    def norm_lambda(self):
        raise NotImplementedError

    def forward(self, g, x):
        x_b = self.batch_norm(x)
        x_g = self.graph_norm(g, x)
        x_a = self.adja_norm(g, x)
        x_n = self.node_norm(x)

        lambda_batch, lambda_graph, lambda_adja, lambda_node = self.norm_lambda() 
        x_new = lambda_batch * x_b + lambda_graph * x_g + lambda_adja * x_a + lambda_node * x_n
        return self.gamma * x_new + self.beta
