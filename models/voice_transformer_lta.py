import torch.nn as nn

class LatentVoiceTransformer(nn.Module):
    def __init__(self, input_dim=6, num_classes=2, latent_dim=32, transformer_dim=64, depth=3, heads=4):
        """
        Args:
            input_dim (int): dimension of input data
            num_classes (int): number of classes
            latent_dim (int): dimension of latent space
            transformer_dim (int): dimension of transformer
            depth (int): number of transformer layers
            heads (int): number of attention heads
        """
        super().__init__()
        # Encoder to a latent representation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        # Project latent space to transformer dimension
        self.latent_to_transformer = nn.Linear(latent_dim, transformer_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim, nhead=heads, dim_feedforward=transformer_dim*2, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.classifier = nn.Sequential(
            nn.LayerNorm(transformer_dim),
            nn.Linear(transformer_dim, num_classes)
        )
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): input data of shape (batch, input_dim)

        Returns:
            torch.Tensor: predicted probabilities of shape (batch, num_classes)
        """
        latent = self.encoder(x)  # (batch, latent_dim)
        transformer_input = self.latent_to_transformer(latent)  # (batch, transformer_dim)
        transformer_input = transformer_input.unsqueeze(1)  # (batch, 1, transformer_dim)
        transformer_output = self.transformer(transformer_input)
        return self.classifier(transformer_output.squeeze(1))
