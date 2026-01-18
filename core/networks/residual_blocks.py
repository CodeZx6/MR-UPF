import torch
import torch.nn as nn
from collections import OrderedDict


def create_conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=True)


class ResidualConvBlock(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.activation = torch.relu
        self.conv = create_conv3x3(filters, filters)
    
    def forward(self, x):
        return self.conv(self.activation(x))


class ResidualUnit(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.block1 = ResidualConvBlock(filters)
        self.block2 = ResidualConvBlock(filters)
    
    def forward(self, x):
        identity = x
        out = self.block1(x)
        out = self.block2(out)
        return out + identity


class StackedResidualUnits(nn.Module):
    def __init__(self, filters, num_units=1):
        super().__init__()
        self.units = nn.Sequential(*[ResidualUnit(filters) for _ in range(num_units)])
    
    def forward(self, x):
        return self.units(x)


class TemporalRecurrentEncoder(nn.Module):
    def __init__(self, channels, seq_len, spatial_dim, num_res_units):
        super().__init__()
        self.seq_len = seq_len
        self.spatial_h = spatial_dim
        self.spatial_w = spatial_dim
        self.gru_cell = nn.GRU(2048, 2048, 2, batch_first=True)
        
        self.channel_adapter = nn.Conv2d(2, channels, 3, 1, 1)
        self.feature_projector = nn.Linear(256*32, 32**2*2, bias=True)
        
        self.residual_pathway = nn.Sequential(OrderedDict([
            ('conv_in', create_conv3x3(in_channels=2, out_channels=128)),
            ('res_blocks', StackedResidualUnits(128, num_res_units)),
            ('dropout', nn.Dropout(0.3)),
            ('activation', nn.ReLU()),
            ('conv_out', create_conv3x3(in_channels=128, out_channels=2))
        ]))
    
    def forward(self, x):
        _, C, H, W = x.size()
        
        encoded_features = self.feature_projector(x.reshape(-1, 256*32))
        encoded_features = encoded_features.reshape(-1, 2, 32, 32)
        encoded_features = self.channel_adapter(encoded_features)
        
        x = encoded_features + x
        
        channel_groups = int(C / 2)
        recurrent_out, hidden_state = self.gru_cell(x.reshape(-1, channel_groups, 2048))
        recurrent_out = hidden_state[-1].reshape(-1, 2, 32, 32)
        
        return self.residual_pathway(recurrent_out)


class SpatialFeatureAggregator(nn.Module):
    def __init__(self, channels, seq_len, num_res_units, spatial_dim, beta):
        super().__init__()
        self.seq_len = seq_len
        self.spatial_h = spatial_dim
        self.spatial_w = spatial_dim
        self.channels = channels
        
        self.channel_groups = int(channels / 2)
        self.adapter_conv = nn.Conv2d(1, self.channel_groups, 3, 1, 1)
        self.feature_linear = nn.Linear(256*128, 32**2*2, bias=True)
        
        self.temporal_branch = TemporalRecurrentEncoder(
            channels=2, 
            seq_len=3, 
            spatial_dim=spatial_dim, 
            num_res_units=num_res_units
        )
        
        self.spatial_branch = nn.Sequential(OrderedDict([
            ('conv_in', create_conv3x3(channels, 128)),
            ('res_units', StackedResidualUnits(128, num_res_units)),
            ('dropout', nn.Dropout(0.5)),
            ('activation', nn.ReLU()),
            ('conv_out', create_conv3x3(128, 2))
        ]))
        
        self.fusion_conv = nn.Conv2d(4, 2, 3, 1, 1)
    
    def forward(self, spatial_features, temporal_features):
        temporal_encoded = self.temporal_branch(temporal_features)
        
        _, C, H, W = spatial_features.size()
        
        spatial_encoded = torch.unsqueeze(spatial_features, dim=1)
        spatial_encoded = self.adapter_conv(spatial_encoded)
        
        spatial_encoded = spatial_encoded.reshape(-1, self.channel_groups, 256*128)
        spatial_encoded = self.feature_linear(spatial_encoded).reshape(-1, self.channels, 32, 32)
        spatial_encoded = self.spatial_branch(spatial_encoded)
        
        fused = spatial_encoded + temporal_encoded
        return fused
