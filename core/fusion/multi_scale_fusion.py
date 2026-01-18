import torch
import torch.nn as nn
import numpy as np
from core.encoders.textual_encoder import TextualEncoder


class MultiScaleSemanticFusion(nn.Module):
    def __init__(self, embed_dim, vocab_size, transformer_width, transformer_heads, transformer_layers):
        super().__init__()
        
        self.spatial_encoder_fine = TextualEncoder(
            embed_dim, 12, vocab_size, transformer_width, transformer_heads, transformer_layers
        )
        self.spatial_encoder_medium = TextualEncoder(
            embed_dim, 12, vocab_size, transformer_width, transformer_heads, transformer_layers
        )
        self.spatial_encoder_coarse = TextualEncoder(
            embed_dim, 12, vocab_size, transformer_width, transformer_heads, transformer_layers
        )
        self.temporal_encoder = TextualEncoder(
            embed_dim, 7, vocab_size, transformer_width, transformer_heads, transformer_layers
        )
        self.context_encoder = TextualEncoder(
            embed_dim, 5, vocab_size, transformer_width, transformer_heads, transformer_layers
        )
        
        self.classifier_binary = nn.Linear(2, 2)
        self.classifier_pentary = nn.Linear(2, 5)
        self.classifier_fine = nn.Linear(2, 143)
        
        self.softmax = nn.Softmax(dim=-1)
        
        self.fusion_conv_binary = nn.Sequential(
            nn.Conv2d(8, 4, 3, 1, 1), nn.ReLU(), nn.Conv2d(4, 2, 3, 1, 1)
        )
        self.fusion_conv_pentary = nn.Sequential(
            nn.Conv2d(8, 4, 3, 1, 1), nn.ReLU(), nn.Conv2d(4, 2, 3, 1, 1)
        )
        self.fusion_conv_fine = nn.Sequential(
            nn.Conv2d(8, 4, 3, 1, 1), nn.ReLU(), nn.Conv2d(4, 2, 3, 1, 1)
        )
        
        self.cross_entropy = nn.CrossEntropyLoss()
        
        self.temporal_projector = nn.Linear(2, 1024)
        self.context_projector = nn.Linear(2, 1024)
        
        self.weight_alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.weight_beta = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.weight_gamma = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        
        self.weight_alpha.data.fill_(0.25)
        self.weight_beta.data.fill_(0.25)
        self.weight_gamma.data.fill_(0.25)
    
    def _process_scale(self, spatial_data, temporal_features, context_features, 
                       spatial_encoder, fusion_conv, classifier, labels):
        batch_size = spatial_data.shape[0]
        spatial_height = spatial_data.shape[-1]
        spatial_width = spatial_data.shape[-2]
        
        spatial_flat = spatial_data.reshape(-1, 2, spatial_height * spatial_width)
        
        spatial_semantic = spatial_encoder(labels.to('cuda')).permute(1, 0).unsqueeze(0)
        spatial_semantic = torch.repeat_interleave(spatial_semantic, repeats=batch_size, dim=0)
        
        fused_spatial = torch.cat((spatial_flat, spatial_semantic), dim=1)
        fused_all = torch.cat((fused_spatial, temporal_features, context_features), dim=1)
        fused_all = fused_all.reshape(-1, 8, spatial_height, spatial_width)
        
        features = fusion_conv(fused_all).reshape(-1, 2, spatial_height * spatial_width)
        features = features + spatial_flat
        
        classification = self.softmax(classifier(features.permute(0, 2, 1)))
        
        label_expanded = labels.reshape(1, spatial_height * spatial_width).to('cuda')
        label_expanded = torch.repeat_interleave(label_expanded, repeats=batch_size, dim=0)
        
        return classification, label_expanded, features
    
    def forward(self, spatial_input, context_input, temporal_tokens, 
                spatial_binary_tokens, spatial_pentary_tokens, spatial_fine_tokens,
                labels_binary, labels_pentary, labels_fine):
        
        spatial_input = torch.reshape(spatial_input, (-1, 2, 32, 32))
        
        context_input = torch.LongTensor(
            np.asarray(context_input.cpu().numpy()[:, :, :, 0])
        ).to('cuda')
        context_input = torch.reshape(context_input.squeeze(-1).permute(0, 2, 1), (-1, 5))
        context_features = self.context_encoder(context_input)
        context_features = self.context_projector(context_features).unsqueeze(1)
        context_features = torch.repeat_interleave(context_features, repeats=spatial_input.shape[1], dim=1)
        
        temporal_tokens = torch.reshape(temporal_tokens, (spatial_input.shape[0], -1)).to('cuda')
        temporal_features = self.temporal_encoder(temporal_tokens)
        temporal_features = self.temporal_projector(temporal_features).unsqueeze(1)
        temporal_features = torch.repeat_interleave(temporal_features, repeats=spatial_input.shape[1], dim=1)
        
        class_binary, labels_binary_exp, feat_binary = self._process_scale(
            spatial_input, temporal_features, context_features,
            self.spatial_encoder_fine, self.fusion_conv_binary, 
            self.classifier_binary, labels_binary
        )
        loss_binary = self.cross_entropy(class_binary.permute(0, 2, 1), labels_binary_exp.long())
        
        class_pentary, labels_pentary_exp, feat_pentary = self._process_scale(
            spatial_input, temporal_features, context_features,
            self.spatial_encoder_medium, self.fusion_conv_pentary, 
            self.classifier_pentary, labels_pentary
        )
        loss_pentary = self.cross_entropy(class_pentary.permute(0, 2, 1), (labels_pentary_exp - 1).long())
        
        class_fine, labels_fine_exp, feat_fine = self._process_scale(
            spatial_input, temporal_features, context_features,
            self.spatial_encoder_coarse, self.fusion_conv_fine, 
            self.classifier_fine, labels_fine
        )
        loss_fine = self.cross_entropy(class_fine.permute(0, 2, 1), (labels_fine_exp - 1).long())
        
        temporal_aggregated = temporal_features + context_features
        
        spatial_fine_embed = self.spatial_encoder_fine(spatial_fine_tokens.to('cuda'))
        spatial_medium_embed = self.spatial_encoder_medium(spatial_pentary_tokens.to('cuda'))
        spatial_coarse_embed = self.spatial_encoder_coarse(spatial_binary_tokens.to('cuda'))
        
        spatial_aggregated = (spatial_fine_embed.reshape(-1, 2, 1024) + 
                             spatial_medium_embed.reshape(-1, 2, 1024) + 
                             spatial_coarse_embed.reshape(-1, 2, 1024))
        
        total_loss = (self.weight_alpha * loss_binary + 
                     self.weight_beta * loss_pentary + 
                     self.weight_gamma * loss_fine)
        
        return temporal_aggregated, spatial_aggregated, total_loss
