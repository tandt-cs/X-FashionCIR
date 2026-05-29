import json
import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from config import Config

class CrossLingualCIREvaluator:
    """
    State-of-the-Art Evaluation Class: Hybrid Spherical-Late Fusion (HSLF) for Zero-shot CIR.
    Implements Riemannian Geometry (Spherical Interpolation) alongside Late Fusion properties.
    """
    def __init__(self):
        print("[*] Initializing Advanced Evaluation Pipeline (HSLF Method)...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[*] Computational backend: {self.device.upper()}")
        
        # Instantiate the Multilingual-CLIP encoder
        self.text_model = SentenceTransformer(Config.TEXT_MODEL_ID, device=self.device)
        
        # Load pre-computed offline features
        print("[*] Loading offline embeddings and index mapping indices...")
        self.image_embeddings = np.load(Config.OUTPUT_EMBEDDINGS) 
        
        with open(Config.OUTPUT_INDEX_MAP, 'r') as f:
            self.id_to_index = json.load(f)
            
        with open(Config.OUTPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            self.val_data = json.load(f)
            
        # Migrate the entire visual corpus embeddings directly to the GPU VRAM
        self.img_tensors = torch.tensor(self.image_embeddings).to(self.device)

    def slerp(self, v0: torch.Tensor, v1: torch.Tensor, t: float, DOT_THRESHOLD: float = 0.9995):
        """
        Spherical Linear Interpolation (SLERP) Operator.
        Interpolates along the surface of a hypersphere to strictly preserve the CLIP manifold.
        """
        # Compute Cosine similarity (Dot Product)
        dot = torch.sum(v0 * v1, dim=-1, keepdim=True)
        dot = torch.clamp(dot, -1.0, 1.0) # Mitigate numerical instability
        
        # Derive the angular distance between vectors
        theta_0 = torch.acos(dot)
        sin_theta_0 = torch.sin(theta_0)
        
        # Compute the interpolated angle proportional to scalar t
        theta_t = theta_0 * t
        sin_theta_t = torch.sin(theta_t)
        
        # Formulate spherical interpolation scaling factors
        s0 = torch.sin(theta_0 - theta_t) / (sin_theta_0 + 1e-8)
        s1 = sin_theta_t / (sin_theta_0 + 1e-8)
        
        # Enforce linear fallback if vectors display collinear properties
        res = torch.where(dot > DOT_THRESHOLD, v0 + t * (v1 - v0), s0 * v0 + s1 * v1)
        return res / res.norm(p=2, dim=-1, keepdim=True)

    def calculate_metrics(self, t_weight: float = 0.5, gamma_weight: float = 0.5) -> dict:
        """
        Quantitative computation utilizing specific hyper-parameters:
        - t_weight (0 -> 1): Spherical interpolation threshold ratio.
        - gamma_weight: The scalar prominence of the Score-level Late Fusion phase.
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
            
            # --- PROPOSED ARCHITECTURAL APPROACH: HYBRID SPHERICAL-LATE FUSION ---
            
            # 1. Early Fusion (Manifold-Preserving): Synthesize optimal hybrid vector via SLERP
            composed_vec = self.slerp(candidate_vec, text_vec, t=t_weight)
            sim_early = torch.matmul(self.img_tensors, composed_vec)
            
            # 2. Late Fusion (Modality Gap Bridge): Measure textual semantic similarity directly against candidates
            sim_late = torch.matmul(self.img_tensors, text_vec)
            
            # 3. Final Ranking: Score-level ensemble
            final_similarities = sim_early + (gamma_weight * sim_late)
            
            # Isolate top-k rank constraints
            _, top50_indices = torch.topk(final_similarities, 50)
            top50_indices = top50_indices.cpu().numpy()
            
            target_idx = self.id_to_index[target_id]
            if target_idx in top50_indices[:10]: correct_at_10 += 1
            if target_idx in top50_indices: correct_at_50 += 1

        r10 = (correct_at_10 / total_queries) * 100
        r50 = (correct_at_50 / total_queries) * 100
        return {"t": t_weight, "gamma": gamma_weight, "R@10": r10, "R@50": r50}

    def tune_hyperparameters(self):
        """Perform a bi-dimensional Grid Search to extrapolate global maxima."""
        t_list = [0.4, 0.5, 0.6, 0.7] # SLERP Ratio
        gamma_list = [0.2, 0.4, 0.6, 0.8] # Late Fusion Scalar Component
        
        print("\n" + "="*60)
        print("INITIATING HSLF HYPER-PARAMETER OPTIMIZATION CAMPAIGN (2D GRID SEARCH)")
        print("="*60)
        
        results = []
        for t in t_list:
            for g in gamma_list:
                res = self.calculate_metrics(t_weight=t, gamma_weight=g)
                results.append(res)
                print(f" [>] Empirical Evaluation: t={t}, γ={g} | Recall@10: {res['R@10']:5.2f}%")
                
        best_cfg = max(results, key=lambda x: x["R@10"])
        
        print("\n" + "*"*60)
        print(" PERFORMANCE EVALUATION METRICS REPORT (NOVELTY HSLF METHOD)")
        print("*"*60)
        for r in results:
            print(f" - SLERP t = {r['t']:<4} | Late Fusion γ = {r['gamma']:<4} | R@10 = {r['R@10']:5.2f}% | R@50 = {r['R@50']:5.2f}%")
        
        print("-" * 60)
        print(f"[🏆] GLOBAL MAXIMA CONFIGURATION: t = {best_cfg['t']}, γ = {best_cfg['gamma']}")
        print(f"Optimal Recall Thresholds: R@10: {best_cfg['R@10']:.2f}% | R@50: {best_cfg['R@50']:.2f}%")
        print("*"*60)

if __name__ == "__main__":
    evaluator = CrossLingualCIREvaluator()
    evaluator.tune_hyperparameters()
