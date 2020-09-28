"""
    File to load dataset based on user control from main file
"""
import torch.nn as nn
from norm.graph_norm import GraphNorm
from norm.adjance_norm import AdjaNodeNorm, AdjaEdgeNorm
from norm.unified_norm import UnifiedNorm

def LoadNorm(NORM_NAME, num_features, is_node=True):
    """
        This function is called in the main.py file 
        returns:
        ; dataset object
    """
    # handling for MNIST or CIFAR Superpixels
    if NORM_NAME == 'BatchNorm':
        return nn.BatchNorm1d(num_features)

    if NORM_NAME == 'LayerNorm':
        return nn.LayerNorm(num_features)

    if NORM_NAME == 'GraphNorm':
        return GraphNorm(num_features, is_node=is_node)

    if NORM_NAME == 'AdjancehNorm':
        if is_node:
            return AdjaNodeNorm(num_features)
        else:
            return AdjaEdgeNorm(num_features)

    if NORM_NAME == 'UnifiedNorm':
        return UnifiedNorm(num_features, is_node)

def normalize(norm, x, g):
    if isinstance(norm, nn.BatchNorm1d) or isinstance(norm, nn.LayerNorm):
        return norm(x)

    if isinstance(norm, GraphNorm) or isinstance(norm, AdjaNodeNorm) or isinstance(norm, AdjaEdgeNorm) or isinstance(norm, UnifiedNorm):
        return norm(g, x)    
