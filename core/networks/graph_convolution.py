import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


def compute_normalized_adjacency(adjacency_matrix):
    degree_vector = np.sum(adjacency_matrix, 0)
    node_count = adjacency_matrix.shape[0]
    degree_matrix = np.zeros((node_count, node_count))
    for idx in range(node_count):
        if degree_vector[idx] > 0:
            degree_matrix[idx, idx] = degree_vector[idx] ** (-1)
    return np.dot(adjacency_matrix, degree_matrix)


class SpectralGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, device):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.zeros((in_features, out_features), 
                       device=torch.device('cuda', 0), 
                       dtype=torch.float32), 
            requires_grad=True
        )
        self.bias = nn.Parameter(
            torch.zeros(out_features, 
                       device=torch.device('cuda', 0), 
                       dtype=torch.float32), 
            requires_grad=True
        )
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, node_features, adjacency):
        transformed = torch.einsum("ijk, kl->ijl", [node_features, self.weight])
        propagated = torch.einsum("ij, kjl->kil", [adjacency, transformed])
        return propagated + self.bias


class HierarchicalGraphNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, device):
        super().__init__()
        self.layer1 = SpectralGraphConvolution(in_dim, hidden_dim, device)
        self.layer2 = SpectralGraphConvolution(hidden_dim, out_dim, device)
    
    def forward(self, features, adjacency):
        h = self.layer1(features, adjacency)
        h = F.relu(h)
        h = self.layer2(h, adjacency)
        return F.relu(h)


class AdaptiveGraphModule(nn.Module):
    def __init__(self, channels, beta_coefficient):
        super().__init__()
        self.beta = beta_coefficient
        self.channels = channels
        self.in_dim = 2
        self.hidden_dim = 512
        self.out_dim = 128
        self.device = torch.device('cuda', 0)
        
        self.graph_net = HierarchicalGraphNetwork(
            in_dim=self.in_dim,
            hidden_dim=self.hidden_dim,
            out_dim=self.out_dim,
            device=self.device
        )
        self.spatial_pooling = nn.MaxPool2d(2, stride=2)
    
    def _construct_adaptive_adjacency(self, attention_maps):
        attn_aggregate = np.sum(attention_maps.cpu().detach().numpy(), axis=0)
        attn_aggregate = np.sum(attn_aggregate, axis=0)
        
        attn_aggregate = attn_aggregate + np.eye(attn_aggregate.shape[0]) * np.max(attn_aggregate)
        threshold = np.abs(np.mean(attn_aggregate)) * (1 + self.beta)
        attn_aggregate = np.where(np.abs(attn_aggregate) < threshold, 0, attn_aggregate)
        
        normalized_adj = compute_normalized_adjacency(attn_aggregate)
        return torch.tensor(normalized_adj, device=self.device, dtype=torch.float32)
    
    def forward(self, spatial_features, attention_qkv):
        adjacency_matrix = self._construct_adaptive_adjacency(attention_qkv)
        
        batch_size = spatial_features.size(0)
        channel_groups = int(self.channels / 2)
        pooled = self.spatial_pooling(spatial_features).reshape(batch_size, channel_groups, 2, 16, 16)
        
        graph_embeddings = None
        for channel_idx in range(pooled.shape[1]):
            channel_data = pooled[:, channel_idx].permute(0, 2, 3, 1).reshape(-1, 256, 2)
            channel_output = self.graph_net(channel_data, adjacency_matrix).reshape(-1, 1, 256, self.out_dim)
            
            if graph_embeddings is None:
                graph_embeddings = channel_output
            else:
                graph_embeddings = torch.cat((channel_output, graph_embeddings), dim=1)
        
        return graph_embeddings
