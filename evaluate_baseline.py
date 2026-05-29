import json
import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from config import Config

class CrossLingualCIREvaluator:
    """
    Lớp đánh giá SOTA: Hybrid Spherical-Late Fusion (HSLF) cho Zero-shot CIR.
    Áp dụng Hình học Riemann (Spherical Interpolation) và Dung hợp Trễ (Late Fusion).
    """
    def __init__(self):
        print("Initializing Advanced Evaluation Pipeline (HSLF Method)...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {self.device.upper()}")
        
        # Load M-CLIP Model
        self.text_model = SentenceTransformer(Config.TEXT_MODEL_ID, device=self.device)
        
        # Load Offline Data
        print("Loading embeddings and index map...")
        self.image_embeddings = np.load(Config.OUTPUT_EMBEDDINGS) 
        
        with open(Config.OUTPUT_INDEX_MAP, 'r') as f:
            self.id_to_index = json.load(f)
            
        with open(Config.OUTPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            self.val_data = json.load(f)
            
        # Đưa toàn bộ vector ảnh lên GPU
        self.img_tensors = torch.tensor(self.image_embeddings).to(self.device)

    def slerp(self, v0: torch.Tensor, v1: torch.Tensor, t: float, DOT_THRESHOLD: float = 0.9995):
        """
        Toán tử SLERP (Spherical Linear Interpolation).
        Nội suy dọc theo bề mặt khối siêu cầu để bảo toàn đa tạp của CLIP.
        """
        # Tính Tích vô hướng (Cosine) giữa 2 vector
        dot = torch.sum(v0 * v1, dim=-1, keepdim=True)
        dot = torch.clamp(dot, -1.0, 1.0) # Tránh lỗi NaN do sai số dấu phẩy động
        
        # Góc giữa 2 vector
        theta_0 = torch.acos(dot)
        sin_theta_0 = torch.sin(theta_0)
        
        # Góc nội suy theo tỷ lệ t
        theta_t = theta_0 * t
        sin_theta_t = torch.sin(theta_t)
        
        # Trọng số nội suy siêu cầu
        s0 = torch.sin(theta_0 - theta_t) / (sin_theta_0 + 1e-8)
        s1 = sin_theta_t / (sin_theta_0 + 1e-8)
        
        # Áp dụng SLERP (Nếu 2 vector quá sát nhau thì lùi về nội suy tuyến tính)
        res = torch.where(dot > DOT_THRESHOLD, v0 + t * (v1 - v0), s0 * v0 + s1 * v1)
        return res / res.norm(p=2, dim=-1, keepdim=True)

    def calculate_metrics(self, t_weight: float = 0.5, gamma_weight: float = 0.5) -> dict:
        """
        Đánh giá với tham số:
        - t_weight (0 -> 1): Tỷ lệ trượt từ Ảnh gốc về phía Text trên khối cầu.
        - gamma_weight: Trọng số của Score-level Late Fusion (Ưu tiên ảnh hưởng trực tiếp của Text).
        """
        correct_at_10, correct_at_50 = 0, 0
        total_queries = len(self.val_data)

        for item in tqdm(self.val_data, desc=f"Evaluating (t={t_weight}, γ={gamma_weight})", leave=False):
            candidate_id, target_id = item['candidate'], item['target']
            
            if candidate_id not in self.id_to_index or target_id not in self.id_to_index:
                continue 
                
            candidate_vec = self.img_tensors[self.id_to_index[candidate_id]]
            
            text_query = " và ".join(item['captions_vn'])
            text_vec = self.text_model.encode(text_query, convert_to_tensor=True, device=self.device)
            text_vec = text_vec / text_vec.norm(p=2, dim=-1, keepdim=True)
            
            # --- CƠ CHẾ ĐỀ XUẤT: HYBRID SPHERICAL-LATE FUSION ---
            
            # 1. Early Fusion (Manifold-Preserving): Dùng SLERP tạo ra vector kết hợp hoàn hảo
            composed_vec = self.slerp(candidate_vec, text_vec, t=t_weight)
            sim_early = torch.matmul(self.img_tensors, composed_vec)
            
            # 2. Late Fusion (Modality Gap Bridge): Tính độ tương đồng trực tiếp của ảnh với câu text
            sim_late = torch.matmul(self.img_tensors, text_vec)
            
            # 3. Final Ranking: Gộp điểm số (Score-level)
            final_similarities = sim_early + (gamma_weight * sim_late)
            
            # Xếp hạng
            _, top50_indices = torch.topk(final_similarities, 50)
            top50_indices = top50_indices.cpu().numpy()
            
            target_idx = self.id_to_index[target_id]
            if target_idx in top50_indices[:10]: correct_at_10 += 1
            if target_idx in top50_indices: correct_at_50 += 1

        r10 = (correct_at_10 / total_queries) * 100
        r50 = (correct_at_50 / total_queries) * 100
        return {"t": t_weight, "gamma": gamma_weight, "R@10": r10, "R@50": r50}

    def tune_hyperparameters(self):
        """Khảo sát Grid Search 2 chiều (t và gamma) để tìm cực trị tối ưu."""
        t_list = [0.4, 0.5, 0.6, 0.7] # Tỷ lệ SLERP (0.5 = 50% Ảnh, 50% Chữ)
        gamma_list = [0.2, 0.4, 0.6, 0.8] # Sức nặng của Late Fusion
        
        print("\n" + "="*60)
        print("KHỞI ĐỘNG CHIẾN DỊCH TỐI ƯU HÓA HSLF (GRID SEARCH 2D)")
        print("="*60)
        
        results = []
        for t in t_list:
            for g in gamma_list:
                res = self.calculate_metrics(t_weight=t, gamma_weight=g)
                results.append(res)
                print(f" > Checked: t={t}, γ={g} | Recall@10: {res['R@10']:5.2f}%")
                
        best_cfg = max(results, key=lambda x: x["R@10"])
        
        print("\n" + "*"*60)
        print(" BÁO CÁO KẾT QUẢ ĐÁNH GIÁ (NOVELTY METHOD: HSLF)")
        print("*"*60)
        for r in results:
            print(f" - SLERP t = {r['t']:<4} | Late Fusion γ = {r['gamma']:<4} | R@10 = {r['R@10']:5.2f}% | R@50 = {r['R@50']:5.2f}%")
        
        print("-" * 60)
        print(f"🏆 CẤU HÌNH SOTA TÌM ĐƯỢC: t = {best_cfg['t']}, γ = {best_cfg['gamma']}")
        print(f"Đạt Recall@10: {best_cfg['R@10']:.2f}% và Recall@50: {best_cfg['R@50']:.2f}%")
        print("*"*60)

if __name__ == "__main__":
    evaluator = CrossLingualCIREvaluator()
    evaluator.tune_hyperparameters()