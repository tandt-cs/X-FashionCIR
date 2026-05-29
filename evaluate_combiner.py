import os
import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import datetime
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from config import Config
from core_models import CombinerNetwork

# ==========================================
# MULTI-METHOD COMPARATIVE EVALUATION PIPELINE
# ==========================================
class ModelEvaluator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[*] Initializing evaluation protocol on architectural backend: {self.device.upper()}")
        
        self.text_model = SentenceTransformer(Config.TEXT_MODEL_ID, device=self.device)
        self.text_model.eval()
        
        print("[*] Retrieving visual embedding spatial matrices...")
        self.img_embeddings = torch.tensor(np.load(Config.OUTPUT_EMBEDDINGS)).to(self.device)
        with open(Config.OUTPUT_INDEX_MAP, 'r') as f:
            self.id_to_idx = json.load(f)
            
        print("[*] Initializing neural integration weights...")
        self.combiner = CombinerNetwork().to(self.device)
        model_path = os.path.join("models", "best_combiner.pth")
        if os.path.exists(model_path):
            self.combiner.load_state_dict(torch.load(model_path, map_location=self.device))
            self.combiner.eval()
        else:
            raise FileNotFoundError(f"[!] Architecture weights absent at {model_path}. Optimization protocol sequence required prior to validation.")

        # Aggregate empirical query dataset
        captions_dir = Config.CAPTIONS_DIR
        val_paths = [os.path.join(captions_dir, f) for f in os.listdir(captions_dir) if f.endswith('.val.vn.json')]
        self.val_data = []
        for vp in val_paths:
            with open(vp, 'r', encoding='utf-8') as f:
                self.val_data.extend(json.load(f))
                
        print(f"[*] Global query resolution space identified: {len(self.val_data)}")

    def compute_metrics(self, similarities, target_idx):
        """Mathematical computation of coverage bounds and absolute spatial ranks."""
        _, top50 = torch.topk(similarities, 50)
        top50_indices = top50.cpu().numpy()
        
        r1 = 1 if target_idx in top50_indices[:1] else 0
        r5 = 1 if target_idx in top50_indices[:5] else 0
        r10 = 1 if target_idx in top50_indices[:10] else 0
        r50 = 1 if target_idx in top50_indices else 0
        
        rank = (similarities > similarities[target_idx]).sum().item() + 1
        return r1, r5, r10, r50, rank

    def run_comparative_evaluation(self):
        print("\n" + "="*70)
        print("🚀 INITIATING MULTI-METHOD SEARCH ARCHITECTURE COMPARISON")
        print("="*70)

        # Baseline and proposed architecture tracking variables
        results = {
            "Image-Only": {"r1": 0, "r5": 0, "r10": 0, "r50": 0, "mr": 0},
            "Text-Only": {"r1": 0, "r5": 0, "r10": 0, "r50": 0, "mr": 0},
            "Vector Addition": {"r1": 0, "r5": 0, "r10": 0, "r50": 0, "mr": 0},
            "Combiner Network": {"r1": 0, "r5": 0, "r10": 0, "r50": 0, "mr": 0}
        }
        
        valid_queries = 0

        with torch.no_grad():
            for item in tqdm(self.val_data, desc="Comprehensive Vector Projection Analysis"):
                cand_id = item.get('candidate')
                target_id = item.get('target')
                
                if not target_id or cand_id not in self.id_to_idx or target_id not in self.id_to_idx:
                    continue
                    
                valid_queries += 1
                target_idx = self.id_to_idx[target_id]
                
                cand_vec = self.img_embeddings[self.id_to_idx[cand_id]].unsqueeze(0)
                text = " và ".join(item['captions_vn'])
                text_vec = self.text_model.encode([text], convert_to_tensor=True, device=self.device)
                text_vec = F.normalize(text_vec, p=2, dim=-1)

                # 1. Visual-Only Isolation (Baseline)
                sim_img = torch.matmul(self.img_embeddings, cand_vec.squeeze())
                m1 = self.compute_metrics(sim_img, target_idx)
                
                # 2. Textual-Only Isolation (Baseline)
                sim_txt = torch.matmul(self.img_embeddings, text_vec.squeeze())
                m2 = self.compute_metrics(sim_txt, target_idx)
                
                # 3. Arithmetic Vector Combination (Baseline)
                add_vec = cand_vec + text_vec
                add_vec = F.normalize(add_vec, p=2, dim=-1)
                sim_add = torch.matmul(self.img_embeddings, add_vec.squeeze())
                m3 = self.compute_metrics(sim_add, target_idx)
                
                # 4. Gated Combiner Sub-network (Proposed Methodology)
                comb_vec = self.combiner(cand_vec, text_vec)
                sim_comb = torch.matmul(self.img_embeddings, comb_vec.squeeze())
                m4 = self.compute_metrics(sim_comb, target_idx)

                # Vector constraint aggregations
                for method, metrics in zip(results.keys(), [m1, m2, m3, m4]):
                    results[method]["r1"] += metrics[0]
                    results[method]["r5"] += metrics[1]
                    results[method]["r10"] += metrics[2]
                    results[method]["r50"] += metrics[3]
                    results[method]["mr"] += metrics[4]

        # Resolution of relative distribution parameters
        for method in results:
            for key in ["r1", "r5", "r10", "r50"]:
                results[method][key] = (results[method][key] / valid_queries) * 100
            results[method]["mr"] = results[method]["mr"] / valid_queries

        self.print_and_save_report(results, valid_queries)

    def print_and_save_report(self, results, total_queries):
        print("\n" + "*"*80)
        print(f" COMPREHENSIVE PERFORMANCE REPORT (GLOBAL VALIDATION MANIFOLD: {total_queries})")
        print("*"*80)
        print(f"{'Methodology':<20} | {'R@1':<7} | {'R@5':<7} | {'R@10':<7} | {'R@50':<7} | {'Mean Rank':<10}")
        print("-" * 80)
        
        for method, metrics in results.items():
            print(f"{method:<20} | {metrics['r1']:5.2f}% | {metrics['r5']:5.2f}% | {metrics['r10']:5.2f}% | {metrics['r50']:5.2f}% | {metrics['mr']:8.1f}")
        print("*"*80)

        # External log file execution mapping
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        log_filename = f"comparison_results_{timestamp}.json"
        
        os.makedirs("results", exist_ok=True)
        log_path = os.path.join("results", log_filename)
        
        log_data = {
            "execution_timestamp": timestamp,
            "validation_instances": total_queries,
            "performance_metrics": results
        }
        
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=4)
        print(f"[*] Statistical deviation benchmarks effectively written to: {log_path}")

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.run_comparative_evaluation()
