import torch.nn as nn
class VoiceTransformer(nn.Module):
    def __init__(self, input_dim=6, num_classes=2, dim=64, depth=3, heads=4):
        """
        Args:
            input_dim (int): Dimensionality of input features
            num_classes (int): Number of classes to predict
            dim (int): Dimensionality of the transformer
            depth (int): Number of transformer layers
            heads (int): Number of attention heads
        """
        super().__init__()
        self.embedding = nn.Linear(input_dim, dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim*2, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass through the VoiceTransformer model.

        Args:
            x (torch.Tensor): Input of shape (batch, input_dim)

        Returns:
            torch.Tensor: Output of shape (batch, num_classes)
        """
        x = self.embedding(x)
        # Simulate a single-token sequence
        x = self.transformer(x.unsqueeze(1))
        return self.classifier(x.squeeze(1))
