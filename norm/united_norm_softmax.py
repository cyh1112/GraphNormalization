import torch.nn as nn
import torch
from norm.united_norm import UnitedNormBase
import torch.nn.functional as F


class UnitedNormSoftmax(UnitedNormBase):

    def __init__(self, *args):
        super(UnitedNormSoftmax, self).__init__(*args)
        self.clamp = False

    def norm_lambda(self):
        concat_lambda = torch.cat([self.lambda_batch.unsqueeze(0), \
                                    self.lambda_graph.unsqueeze(0), \
                                    self.lambda_adja.unsqueeze(0), \
                                    self.lambda_node.unsqueeze(0)], dim=0)

        softmax_lambda = F.softmax(concat_lambda, dim=0)
        return softmax_lambda[0], softmax_lambda[1], softmax_lambda[2], softmax_lambda[3] 