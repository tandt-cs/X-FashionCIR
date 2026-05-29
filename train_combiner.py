import os
import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import datetime
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from config import Config
from core_models import CombinerNetwork

# ==========================================
# XỬ LÝ DỮ LIỆU ĐA TẬP
# ==========================================
class FashionIQDataset(Dataset):
    def __init__(self, json_paths):
        self.data = []
        for path in json_paths:
            with open(path, 'r', encoding='utf-8') as f:
                self.data.extend(json.load(f))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item['candidate'], item['target'], " và ".join(item['captions_vn'])

# ==========================================
# 3. QUY TRÌNH HUẤN LUYỆN VÀ LƯU NHẬT KÝ
# ==========================================
class CombinerTrainer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[#] Khởi tạo quá trình huấn luyện trên thiết bị: {self.device.upper()}")
        
        self.text_model = SentenceTransformer(Config.TEXT_MODEL_ID, device=self.device)
        self.text_model.eval()
        
        self.combiner = CombinerNetwork().to(self.device)
        
        print("[#] Đang tải ma trận vectơ hình ảnh...")
        self.img_embeddings = torch.tensor(np.load(Config.OUTPUT_EMBEDDINGS)).to(self.device)
        with open(Config.OUTPUT_INDEX_MAP, 'r') as f:
            self.id_to_idx = json.load(f)
            
        # Siêu tham số
        self.epochs = 30
        self.batch_size = 128
        self.temperature = 0.05
        self.margin = 0.2        
        
        self.optimizer = torch.optim.AdamW(self.combiner.parameters(), lr=2e-4, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        
        # Thu thập dữ liệu huấn luyện và kiểm định
        captions_dir = Config.CAPTIONS_DIR
        all_files = os.listdir(captions_dir)
        
        train_paths = [os.path.join(captions_dir, f) for f in all_files if f.endswith('.train.vn.json')]
        val_paths = [os.path.join(captions_dir, f) for f in all_files if f.endswith('.val.vn.json')]
        
        if not train_paths or not val_paths:
            raise Exception("Không tìm thấy đủ tập huấn luyện hoặc kiểm định. Vui lòng kiểm tra lại dữ liệu.")
            
        print(f"[#] Tổng hợp dữ liệu: {len(train_paths)} tệp huấn luyện và {len(val_paths)} tệp kiểm định.")
            
        self.train_loader = DataLoader(FashionIQDataset(train_paths), batch_size=self.batch_size, shuffle=True, drop_last=True)
        
        self.val_data = []
        for vp in val_paths:
            with open(vp, 'r', encoding='utf-8') as f:
                self.val_data.extend(json.load(f))

    def compute_loss(self, query_feats, target_feats):
        sim_matrix = torch.matmul(query_feats, target_feats.T)
        
        logits = sim_matrix / self.temperature
        labels = torch.arange(len(logits), device=self.device)
        loss_infonce = F.cross_entropy(logits, labels)
        
        positives = torch.diag(sim_matrix)
        mask = torch.eye(len(sim_matrix), dtype=torch.bool, device=self.device)
        negatives_matrix = sim_matrix.masked_fill(mask, -float('inf'))
        hardest_negatives, _ = negatives_matrix.max(dim=1)
        
        loss_triplet = F.relu(hardest_negatives - positives + self.margin).mean()
        
        return loss_infonce + 0.2 * loss_triplet

    def evaluate(self, data, desc="Đánh giá"):
        if not data:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        self.combiner.eval()
        correct_at_1, correct_at_5, correct_at_10, correct_at_50 = 0, 0, 0, 0
        mean_rank = 0.0
        total = 0
        
        with torch.no_grad():
            for item in tqdm(data, desc=desc, leave=False):
                cand_id = item.get('candidate')
                target_id = item.get('target')
                
                if not target_id or cand_id not in self.id_to_idx or target_id not in self.id_to_idx:
                    continue
                    
                total += 1
                cand_vec = self.img_embeddings[self.id_to_idx[cand_id]].unsqueeze(0)
                text = " và ".join(item['captions_vn'])
                text_vec = self.text_model.encode([text], convert_to_tensor=True, device=self.device)
                text_vec = F.normalize(text_vec, p=2, dim=-1)
                
                query_vec = self.combiner(cand_vec, text_vec)
                sims = torch.matmul(self.img_embeddings, query_vec.squeeze())
                _, top50 = torch.topk(sims, 50)
                top50_indices = top50.cpu().numpy()
                
                target_idx = self.id_to_idx[target_id]
                
                # Tính toán thứ hạng trung bình (Mean Rank)
                # Đếm số lượng ảnh có điểm tương đồng lớn hơn ảnh mục tiêu
                rank = (sims > sims[target_idx]).sum().item() + 1
                mean_rank += rank
                
                # Cập nhật thuật toán tính toán đa mức độ phủ
                if target_idx in top50_indices[:1]:
                    correct_at_1 += 1
                if target_idx in top50_indices[:5]:
                    correct_at_5 += 1
                if target_idx in top50_indices[:10]:
                    correct_at_10 += 1
                if target_idx in top50_indices:
                    correct_at_50 += 1
                    
        if total == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0
            
        r1 = (correct_at_1 / total) * 100
        r5 = (correct_at_5 / total) * 100
        r10 = (correct_at_10 / total) * 100
        r50 = (correct_at_50 / total) * 100
        mr = mean_rank / total
        
        return r1, r5, r10, r50, mr

    def run(self):
        print("\n" + "="*60)
        print("🚀 BẮT ĐẦU QUY TRÌNH HUẤN LUYỆN")
        print("="*60)
        
        best_val_r1 = 0.0
        best_val_r5 = 0.0
        best_val_r10 = 0.0
        best_val_r50 = 0.0
        best_val_mr = 0.0
        best_epoch = 0
        os.makedirs("models", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        save_path = os.path.join("models", "best_combiner.pth")

        for epoch in range(self.epochs):
            self.combiner.train()
            total_loss = 0
            
            pbar = tqdm(self.train_loader, desc=f"Chu kỳ {epoch+1}/{self.epochs}")
            for batch in pbar:
                cand_ids, target_ids, texts = batch
                
                valid_mask = [c in self.id_to_idx and t in self.id_to_idx for c, t in zip(cand_ids, target_ids)]
                if not any(valid_mask): continue
                
                cand_ids = [c for c, m in zip(cand_ids, valid_mask) if m]
                target_ids = [t for t, m in zip(target_ids, valid_mask) if m]
                texts = [t for t, m in zip(texts, valid_mask) if m]
                
                cand_vecs = torch.stack([self.img_embeddings[self.id_to_idx[c]] for c in cand_ids])
                target_vecs = torch.stack([self.img_embeddings[self.id_to_idx[t]] for t in target_ids])
                
                with torch.no_grad():
                    text_vecs = self.text_model.encode(texts, convert_to_tensor=True, device=self.device)
                    text_vecs = F.normalize(text_vecs, p=2, dim=-1)
                
                self.optimizer.zero_grad()
                query_vecs = self.combiner(cand_vecs, text_vecs)
                
                loss = self.compute_loss(query_vecs, target_vecs)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
            self.scheduler.step()
            avg_loss = total_loss / len(self.train_loader)
            
            val_r1, val_r5, val_r10, val_r50, val_mr = self.evaluate(self.val_data, desc="Đang kiểm định")
            
            print(f"\n[Chu kỳ {epoch+1}] Sai số: {avg_loss:.4f} | R@1: {val_r1:.2f}% | R@5: {val_r5:.2f}% | R@10: {val_r10:.2f}% | R@50: {val_r50:.2f}% | Mean Rank: {val_mr:.1f}")
            
            # Luôn sử dụng Recall@10 làm chỉ số quyết định để lưu mô hình (Chuẩn Benchmark Fashion-IQ)
            if val_r10 > best_val_r10:
                best_val_r1 = val_r1
                best_val_r5 = val_r5
                best_val_r10 = val_r10
                best_val_r50 = val_r50
                best_val_mr = val_mr
                best_epoch = epoch + 1
                torch.save(self.combiner.state_dict(), save_path)
                print(f"🔥 Cập nhật trọng số tối ưu tại: {save_path}")

        print("\n" + "*"*60)
        print(" TỔNG HỢP KẾT QUẢ HUẤN LUYỆN")
        print("*"*60)
        print(f" => Chu kỳ đạt đỉnh       : {best_epoch}")
        print(f" => Độ phủ tại mức 1      : {best_val_r1:.2f}%")
        print(f" => Độ phủ tại mức 5      : {best_val_r5:.2f}%")
        print(f" => Độ phủ tại mức 10     : {best_val_r10:.2f}%")
        print(f" => Độ phủ tại mức 50     : {best_val_r50:.2f}%")
        print(f" => Thứ hạng trung bình   : {best_val_mr:.1f}")
        print("*"*60)
        
        self.save_evaluation_log(best_epoch, best_val_r1, best_val_r5, best_val_r10, best_val_r50, best_val_mr)

    def save_evaluation_log(self, best_epoch, r1, r5, r10, r50, mr):
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        log_filename = f"results_{timestamp}.json"
        log_path = os.path.join("results", log_filename)
        
        log_data = {
            "thoi_gian_hoan_tat": timestamp,
            "chu_ky_toi_uu": best_epoch,
            "chi_so_do_phu_muc_1": round(r1, 2),
            "chi_so_do_phu_muc_5": round(r5, 2),
            "chi_so_do_phu_muc_10": round(r10, 2),
            "chi_so_do_phu_muc_50": round(r50, 2),
            "thu_hang_trung_binh_mr": round(mr, 2),
            "sieu_tham_so": {
                "tong_so_chu_ky": self.epochs,
                "kich_thuoc_bo_du_lieu": self.batch_size,
                "nhiet_do_tuong_phan": self.temperature,
                "bien_do_hinh_phat": self.margin
            }
        }
        
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=4)
        print(f"Toàn bộ dữ liệu đánh giá đã được trích xuất thành công vào: {log_path}")

if __name__ == "__main__":
    trainer = CombinerTrainer()
    trainer.run()