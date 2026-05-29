import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# KIẾN TRÚC LÕI: MẠNG NƠ-RON KẾT HỢP
# ==========================================
class CombinerNetwork(nn.Module):
    """
    Kiến trúc mạng dung hợp đặc trưng dựa trên cơ chế cổng.
    Tiếp nhận vectơ hình ảnh và vectơ văn bản, kết xuất vectơ truy vấn tối ưu.
    """
    def __init__(self, embed_dim=512, hidden_dim=1024):
        super().__init__()
        # Tầng dung hợp phi tuyến tính
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embed_dim)
        )
        # Tầng quyết định giữ lại hoặc loại bỏ đặc trưng
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )

    def forward(self, img_feat, text_feat):
        # Nối ma trận đặc trưng
        combined = torch.cat([img_feat, text_feat], dim=-1)
        
        # Khởi tạo đặc trưng lai
        fused_feat = self.fusion(combined)
        
        # Tính toán ma trận trọng số cổng
        g = self.gate(combined)
        
        # Dung hợp có trọng số
        out = g * img_feat + (1 - g) * fused_feat
        
        # Chuẩn hóa bề mặt đa tạp
        return F.normalize(out, p=2, dim=-1)