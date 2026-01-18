import torch
import torch.nn as nn
from collections import OrderedDict


class CrossModalAttentionBlock(nn.Module):
    def __init__(self, dim=1024, num_heads=4, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.scale = dim ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(dim, dim)
        self.projection_dropout = nn.Dropout(dropout)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, 1024), requires_grad=False)
        
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
    
    def forward(self, temporal_context, spatial_flow):
        temporal_context = temporal_context.reshape(-1, 20, 1024) + self.pos_embed
        spatial_flow = spatial_flow.reshape(-1, 20, 1024) + self.pos_embed
        
        q = self.query_proj(spatial_flow).transpose(-2, -1)
        k = self.key_proj(temporal_context).transpose(-2, -1)
        v = self.value_proj(spatial_flow).transpose(-2, -1)
        
        attention_scores = (q @ k.transpose(-2, -1)) * self.scale
        attention_weights = attention_scores.softmax(dim=-1)
        
        attended = (attention_weights @ v).transpose(-2, -1)
        output = self.projection(attended)
        output = self.projection_dropout(output)
        
        return output


class HierarchicalSpatioTemporalModel(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, base_channels=64,
                 img_height=32, img_width=32, beta_coeff=0.5):
        super().__init__()
        self.num_residual_units = 4
        self.img_h = img_height
        self.img_w = img_width
        
        self.activation_fn = torch.tanh
        
        from core.networks.residual_blocks import SpatialFeatureAggregator
        from core.fusion.multi_scale_fusion import MultiScaleSemanticFusion
        
        self.spatial_temporal_branch = SpatialFeatureAggregator(
            channels=20,
            seq_len=3,
            num_res_units=self.num_residual_units,
            spatial_dim=self.img_h,
            beta=beta_coeff
        )
        
        self.semantic_fusion_module = MultiScaleSemanticFusion(
            embed_dim=2,
            vocab_size=50000,
            transformer_width=16,
            transformer_heads=4,
            transformer_layers=3
        )
        
        self.channel_reduction = nn.Sequential(
            nn.Conv2d(40, 32, 3, 1, 1), 
            nn.ReLU(),
            nn.Conv2d(32, 20, 3, 1, 1), 
            nn.ReLU()
        )
        
        self.cross_attention_block = CrossModalAttentionBlock()
    
    def _reshape_features(self, feature_tensor, target_shape):
        feature_tensor = feature_tensor.permute(0, 2, 1)
        return feature_tensor.reshape(target_shape.shape[0], -1, 
                                     target_shape.shape[-2], 
                                     target_shape.shape[-1])
    
    def _concatenate_modalities(self, temporal_feats, spatial_feats, reference):
        return torch.concat(
            (self._reshape_features(temporal_feats, reference), 
             self._reshape_features(spatial_feats, reference)), 
            dim=1
        )
    
    def forward(self, temporal_input, context_input, text_tokens, 
                spatial_binary_tokens, spatial_pentary_tokens, spatial_fine_tokens,
                labels_binary, labels_pentary, labels_fine):
        
        temporal_embed, spatial_embed, semantic_loss = self.semantic_fusion_module(
            temporal_input, context_input, text_tokens,
            spatial_binary_tokens, spatial_pentary_tokens, spatial_fine_tokens,
            labels_binary, labels_pentary, labels_fine
        )
        
        seq_len = temporal_input.shape[1]
        num_channels = temporal_input.shape[2]
        
        temporal_sequence = temporal_input.view(-1, seq_len * num_channels, self.img_h, self.img_w)
        
        temporal_embed_reshaped = temporal_embed.view(-1, num_channels, self.img_h, self.img_w)
        temporal_embed_reshaped = temporal_embed_reshaped.view(-1, seq_len * num_channels, self.img_h, self.img_w)
        
        spatial_embed_reshaped = spatial_embed.view(-1, num_channels, self.img_h, self.img_w)
        spatial_embed_reshaped = spatial_embed_reshaped.view(-1, seq_len * num_channels, self.img_h, self.img_w)
        
        combined_spatial = torch.cat((spatial_embed_reshaped, temporal_sequence), dim=1)
        combined_spatial = self.channel_reduction(combined_spatial)
        
        attended_features = self.cross_attention_block(temporal_embed_reshaped, combined_spatial)
        attended_features = attended_features.reshape(temporal_sequence.shape) + temporal_sequence
        
        final_prediction = self.spatial_temporal_branch(attended_features, temporal_input)
        
        return final_prediction, semantic_loss
