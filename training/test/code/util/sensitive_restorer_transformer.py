"""
Transformer-based Sensitive Channel Restorer
专注于最大化容错能力的架构设计
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
import math


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力机制"""
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        """
        x: [B, seq_len, embed_dim]
        """
        B, seq_len, embed_dim = x.shape
        residual = x
        
        # Multi-head attention
        q = self.q_proj(x).view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, D]
        k = self.k_proj(x).view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)  # [B, H, L, D]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, seq_len, embed_dim)
        output = self.out_proj(attn_output)
        
        # Residual connection and layer norm
        output = self.norm(output + residual)
        return output


class TransformerBlock(nn.Module):
    """Transformer块：自注意力 + FFN"""
    def __init__(self, embed_dim, num_heads=8, ffn_dim=None, dropout=0.1):
        super().__init__()
        if ffn_dim is None:
            ffn_dim = embed_dim * 4
        
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # Self-attention
        x = self.attention(x)
        # FFN with residual
        residual = x
        x = self.ffn(x)
        x = self.norm(x + residual)
        return x


class LayerWiseFeatureProcessor(nn.Module):
    """
    Processes a list of layer-wise features. Each layer's feature tensor
    is processed by its own small MLP, and then combined with a learned
    layer embedding.
    """
    def __init__(self, num_layers, feature_dims_per_layer, embed_dim):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        
        # Create a specific projection MLP for each layer's feature dimension
        self.layer_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            ) for feat_dim in feature_dims_per_layer
        ])
        
        # Learned embeddings for each layer position
        self.layer_embeddings = nn.Embedding(num_layers, embed_dim)

    def forward(self, layer_features: List[torch.Tensor]):
        """
        Args:
            layer_features: A list of tensors, where layer_features[i] is
                            the feature tensor for the i-th sensitive layer.
                            Shape of each tensor: [B, feature_dim_i]
        
        Returns:
            A tensor of shape [B, num_layers, embed_dim] ready for the Transformer.
        """
        B = layer_features[0].shape[0]
        layer_embeds = []
        
        for i, feat in enumerate(layer_features):
            # Process with layer-specific MLP
            proj_feat = self.layer_projections[i](feat)  # [B, embed_dim]
            
            # Add positional (layer) embedding
            layer_id = torch.tensor(i, device=feat.device).expand(B)
            layer_emb = self.layer_embeddings(layer_id)  # [B, embed_dim]
            
            layer_embeds.append(proj_feat + layer_emb)
        
        # Stack to create the sequence for the transformer
        x = torch.stack(layer_embeds, dim=1)  # [B, num_layers, embed_dim]
        return x

class TransformerRestorer(nn.Module):
    def __init__(self, num_layers: int, feature_dims_per_layer: List[int], num_classes: int, embed_dim: int, num_transformer_layers: int, num_heads: int):
        super().__init__()
        
        self.feature_processor = LayerWiseFeatureProcessor(
            num_layers, feature_dims_per_layer, embed_dim
        )
        
        # Learnable token that will act as the "summary" of all layer features
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        self.detector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.restorer = nn.Sequential(
            nn.Linear(embed_dim + num_classes, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, logits, layer_features: List[torch.Tensor]):
        B = logits.shape[0]
        
        # 1. Process layer-wise features independently
        x = self.feature_processor(layer_features)  # [B, num_layers, embed_dim]
        
        # 2. Prepend the [CLS] token to the sequence
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, num_layers + 1, embed_dim]
        
        # 3. Global context fusion with Transformer
        x = self.transformer_encoder(x)  # [B, num_layers + 1, embed_dim]
        
        # 4. Use the output of the [CLS] token as the aggregated feature representation
        cls_output = x[:, 0]  # [B, embed_dim]
        
        # 5. Gating and Restoration based on the [CLS] token's representation
        # Temporarily disable the gating mechanism to diagnose learning issues.
        # Force the restorer to always apply its correction.
        # gate = self.detector(cls_output)
        
        augmented = torch.cat([cls_output, logits], dim=1)
        delta = self.restorer(augmented)
        
        # Unconditionally apply the delta
        restored = logits + delta
        
        # Return a dummy gate value for logging purposes
        gate = torch.tensor(1.0, device=logits.device).expand(B, 1)
        
        return restored, gate


class LayerwiseRestorer(nn.Module):
    """
    A restorer model that uses a separate small MLP "expert" for each sensitive layer's features.
    This "divide and conquer" approach simplifies the learning task for each expert.
    """
    def __init__(self, num_layers: int, feature_dims_per_layer: List[int], num_classes: int, expert_hidden_dim: int):
        super().__init__()
        self.num_layers = num_layers
        
        self.experts = nn.ModuleList()
        for feature_dim in feature_dims_per_layer:
            # Each expert now takes both layer-specific features AND the faulted logits as input
            input_dim = feature_dim + num_classes
            
            expert = nn.Sequential(
                nn.Linear(input_dim, expert_hidden_dim),
                nn.LayerNorm(expert_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(expert_hidden_dim, expert_hidden_dim // 2),
                nn.LayerNorm(expert_hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(expert_hidden_dim // 2, num_classes)
            )
            self.experts.append(expert)

    def forward(self, logits, layer_features: List[torch.Tensor]):
        """
        Args:
            logits (torch.Tensor): The faulted logits from the main model.
            layer_features (List[torch.Tensor]): A list of feature tensors, one from each sensitive layer.
        
        Returns:
            torch.Tensor: The restored logits.
            torch.Tensor: A dummy gate tensor for logging compatibility.
        """
        B = logits.shape[0]
        
        all_deltas = []
        for i, features in enumerate(layer_features):
            # Concatenate local features with global faulted logits to provide full context
            expert_input = torch.cat([features, logits], dim=1)
            
            # Each expert produces a delta suggestion based on its layer's features and the global logits
            delta_i = self.experts[i](expert_input)
            all_deltas.append(delta_i)
        
        # Aggregate deltas from all experts, for example, by averaging
        if not all_deltas:
            # If for some reason there are no features, return original logits
            return logits, torch.tensor(0.0, device=logits.device).expand(B, 1)

        # Stack and then average the delta suggestions
        stacked_deltas = torch.stack(all_deltas, dim=0) # Shape: [num_layers, B, num_classes]
        mean_delta = torch.mean(stacked_deltas, dim=0) # Shape: [B, num_classes]
        
        # Apply the aggregated correction
        # We are still forcing the correction to be applied, as the gating mechanism was problematic
        restored = logits + mean_delta
        
        # For compatibility with existing logging, return a dummy gate value of 1.0
        gate = torch.tensor(1.0, device=logits.device).expand(B, 1)
        
        return restored, gate

