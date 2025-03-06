import torch.nn as nn
class VoiceTransformer(nn.Module):
    def __init__(self, input_dim=6, num_classes=2, dim=64, depth=3, heads=4):
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
        x = self.embedding(x)
        # Simulate a single-token sequence
        x = self.transformer(x.unsqueeze(1))
        return self.classifier(x.squeeze(1))