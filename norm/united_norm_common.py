import torch.nn as nn
import torch
from norm.united_norm import UnitedNormBase


class UnitedNormCommon(UnitedNormBase):

    def __init__(self, *args):
        super(UnitedNormCommon, self).__init__(*args)
        self.clamp = True

    def norm_lambda(self):
        lambda_sum = self.lambda_batch + self.lambda_graph + self.lambda_adja + self.lambda_node
        return self.lambda_batch / lambda_sum, \
                self.lambda_graph / lambda_sum, \
                self.lambda_adja / lambda_sum, \
                self.lambda_node / lambda_sum