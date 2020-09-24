import torch.nn as nn
import torch


class GraphNorm(nn.Module):
    """
        Param: []
    """
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.num_features = num_features
        self.gamma = nn.Parameter(torch.ones(self.num_features))
        self.beta = nn.Parameter(torch.zeros(self.num_features))

    def norm(self, x):
        mean = x.mean(dim = 0, keepdim = True)
        var = x.std(dim = 0, keepdim = True)

        x = (x - mean) / (var + self.eps)
        return x

    def forward(self, x, graph_size):
        x_list = torch.split(x, graph_size)
        norm_list = []
        for x in x_list:
            norm_list.append(self.norm(x))

        x = torch.cat(norm_list, 0)
        return self.gamma * x + self.beta


