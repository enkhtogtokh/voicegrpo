import torch.nn as nn
import torch.nn.functional as F
import torch

class VoiceMoETransformer(nn.Module):
    def __init__(self, input_dim=6, num_classes=2, dim=64, depth=3, heads=4, num_experts=3):
        """
        Args:
            input_dim (int): Dimensionality of input features
            num_classes (int): Number of classes to predict
            dim (int): Dimensionality of the transformer (and experts)
            depth (int): Number of transformer layers (per expert)
            heads (int): Number of attention heads
            num_experts (int): Number of experts to use
        """
        super().__init__()
        self.embedding = nn.Linear(input_dim, dim)
        self.num_experts = num_experts
        
        # Each expert is its own transformer encoder (with 'depth' layers)
        expert_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim*2, dropout=0.1
        )
        self.experts = nn.ModuleList([
            nn.TransformerEncoder(expert_layer, num_layers=depth) for _ in range(num_experts)
        ])
        # Gating network to combine expert outputs
        self.gate = nn.Linear(dim, num_experts)
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass through the MoE transformer model.

        Args:
            x (torch.Tensor): Input of shape (batch, input_dim)

        Returns:
            torch.Tensor: Output of shape (batch, num_classes)
        """
        x = self.embedding(x)  # (batch, dim)
        x_seq = x.unsqueeze(1)  # (batch, 1, dim)
        expert_outputs = [expert(x_seq) for expert in self.experts]  # each: (batch, 1, dim)
        gate_weights = F.softmax(self.gate(x), dim=-1)  # (batch, num_experts)
        stacked = torch.stack([out.squeeze(1) for out in expert_outputs], dim=1)  # (batch, num_experts, dim)
        combined = (gate_weights.unsqueeze(-1) * stacked).sum(dim=1)  # (batch, dim)
        return self.classifier(combined)
