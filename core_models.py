import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# CORE ARCHITECTURE: COMBINER NETWORK
# ==========================================
class CombinerNetwork(nn.Module):
    """
    Feature fusion network architecture based on a non-linear gating mechanism.
    Receives visual and textual embeddings to compute an optimal composed query vector.
    """
    def __init__(self, embed_dim=512, hidden_dim=1024):
        super().__init__()
        # Non-linear feature fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embed_dim)
        )
        # Gating layer to dynamically determine the retention of visual features
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )

    def forward(self, img_feat, text_feat):
        # Concatenate multi-modal feature matrices
        combined = torch.cat([img_feat, text_feat], dim=-1)
        
        # Initialize the hybrid feature representation
        fused_feat = self.fusion(combined)
        
        # Compute the gate weight matrix
        g = self.gate(combined)
        
        # Execute weighted fusion via element-wise multiplication
        out = g * img_feat + (1 - g) * fused_feat
        
        # Perform L2 normalization to align with the hyperspherical manifold
        return F.normalize(out, p=2, dim=-1)
