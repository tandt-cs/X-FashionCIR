import os
import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from PIL import Image

# --- Suppress dependency warnings to maintain a clean execution terminal ---
import warnings
warnings.filterwarnings("ignore")
import transformers
transformers.logging.set_verbosity_error()

from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer

from config import Config

# ==========================================
# 1. CORE NETWORK ARCHITECTURE RECONSTRUCTION
# ==========================================
class CombinerNetwork(nn.Module):
    def __init__(self, embed_dim=512, hidden_dim=1024):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )

    def forward(self, img_feat, text_feat):
        combined = torch.cat([img_feat, text_feat], dim=-1)
        fused_feat = self.fusion(combined)
        g = self.gate(combined)
        out = g * img_feat + (1 - g) * fused_feat
        return F.normalize(out, p=2, dim=-1)

# ==========================================
# 2. SYSTEM RAM INITIALIZATION (CACHED)
# ==========================================
@st.cache_resource
def load_system():
    """Load foundational models and offline database into RAM for seamless latency-free inference."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Loading inference systems on computational backend: {device.upper()}...")
    
    # 1. Instantiate the M-CLIP Text Encoder
    text_model = SentenceTransformer(Config.TEXT_MODEL_ID, device=device)
    
    # 2. Instantiate the CLIP Vision Model
    vision_model = CLIPModel.from_pretrained(Config.VISION_MODEL_ID).to(device).eval()
    vision_processor = CLIPProcessor.from_pretrained(Config.VISION_MODEL_ID)
    
    # 3. Instantiate the empirically trained Combiner Network
    combiner = CombinerNetwork().to(device)
    combiner_path = os.path.join("models", "best_combiner.pth")
    if os.path.exists(combiner_path):
        combiner.load_state_dict(torch.load(combiner_path, map_location=device))
        print("[+] Optimal combiner weights successfully mapped into memory.")
    else:
        st.warning("Combiner weights missing! System fallback to stochastic parameters initialized.")
    combiner.eval()
    
    # 4. Load the comprehensive offline representation database (approx. 77,000 artifacts)
    embeddings = np.load(Config.OUTPUT_EMBEDDINGS)
    img_tensors = torch.tensor(embeddings).to(device)
    
    with open(Config.OUTPUT_INDEX_MAP, 'r') as f:
        id_to_index = json.load(f)
        index_to_id = {v: k for k, v in id_to_index.items()}
        
    return text_model, vision_model, vision_processor, combiner, img_tensors, index_to_id, device

# ==========================================
# 3. STREAMLIT INTERFACE (MATERIAL DESIGN)
# ==========================================
def main():
    st.set_page_config(page_title="V-Fashion CIR", layout="wide", page_icon="🛍️")
    
    # --- CSS INJECTION ADHERING TO GOOGLE MATERIAL DESIGN STANDARDS ---
    st.markdown("""
        <style>
        /* Typography and Neutral Background Layout */
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
        html, body, [class*="css"] {
            font-family: 'Roboto', sans-serif;
            background-color: #f8f9fa;
        }
        
        /* Central Header Formatting */
        .main-header {
            text-align: center;
            padding: 2rem 0 3rem 0;
            color: #202124;
        }
        .main-header h1 {
            font-weight: 700;
            letter-spacing: -0.5px;
            margin-bottom: 0.5rem;
        }
        .main-header p {
            color: #5f6368;
            font-size: 1.1rem;
        }

        /* Material Component Button Constraints */
        div.stButton > button {
            background-color: #1a73e8 !important; /* Google Blue */
            color: white !important;
            border-radius: 24px !important; /* Pill shape structure */
            border: none !important;
            box-shadow: 0 2px 4px rgba(26, 115, 232, 0.3) !important;
            font-weight: 500 !important;
            padding: 0.5rem 2rem !important;
            transition: all 0.3s ease !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            width: 100%;
        }
        div.stButton > button:hover {
            background-color: #1557b0 !important;
            box-shadow: 0 4px 8px rgba(26, 115, 232, 0.4) !important;
            transform: translateY(-1px);
        }

        /* Material Layout Artifacts */
        [data-testid="stImage"] {
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
            transition: all 0.3s cubic-bezier(.25,.8,.25,1);
            cursor: pointer;
            margin-bottom: 0.5rem;
        }
        [data-testid="stImage"]:hover {
            transform: scale(1.03);
            box-shadow: 0 14px 28px rgba(0,0,0,0.15), 0 10px 10px rgba(0,0,0,0.12);
            z-index: 10;
        }

        /* Captions Optimization */
        .result-caption {
            text-align: center;
            font-size: 0.9rem;
            color: #3c4043;
            background: white;
            padding: 0.5rem;
            border-radius: 8px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
            margin-top: -5px;
            margin-bottom: 1.5rem;
        }
        .score-badge {
            background: #e8f0fe;
            color: #1a73e8;
            padding: 2px 8px;
            border-radius: 12px;
            font-weight: 500;
            font-size: 0.85rem;
        }
        
        /* Control Panel Artifacts */
        .control-panel {
            background: white;
            padding: 2rem;
            border-radius: 16px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
            border: 1px solid #f1f3f4;
        }
        
        /* Viewport Boundaries */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="main-header">
            <h1>🛍️ CS2224 - Tìm Kiếm Thời Trang Đa Phương Thức</h1>
            <p>Tải lên ảnh mẫu và mô tả điều bạn muốn thay đổi bằng ngôn ngữ tự nhiên.</p>
        </div>
    """, unsafe_allow_html=True)
    
    try:
        text_model, vision_model, vision_processor, combiner, img_tensors, index_to_id, device = load_system()
    except Exception as e:
        st.error(f"Lỗi khởi tạo hệ thống: {e}")
        st.stop()
        
    # Layout Distribution Architecture
    col1, col2 = st.columns([1.2, 2.8], gap="large")
    
    with col1:
        st.markdown('<div class="control-panel">', unsafe_allow_html=True)
        st.subheader("📝 Bảng điều khiển")
        
        uploaded_file = st.file_uploader("1. Tải lên ảnh tham chiếu (JPG/PNG):", type=["jpg", "jpeg", "png"])
        
        ref_image = None
        if uploaded_file is not None:
            ref_image = Image.open(uploaded_file).convert("RGB")
            # Apply geometric constraints to UI visuals
            st.image(ref_image, caption="Ảnh gốc (Click để phóng to)", width=280)
            st.markdown("<br>", unsafe_allow_html=True)
            
        modifier_text = st.text_area(
            "2. Yêu cầu tinh chỉnh (Tiếng Việt):", 
            placeholder="Ví dụ: Đổi sang màu đỏ, thiết kế tay dài và cổ tim...",
            height=120
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        search_btn = st.button("🔍 Tiến hành Tìm kiếm", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.subheader("✨ Kết quả gợi ý")
        st.markdown("<hr style='margin-top: 0; border-color: #e0e0e0;'>", unsafe_allow_html=True)
        
        if search_btn:
            if not ref_image:
                st.warning("Vui lòng tải lên một bức ảnh tham chiếu!")
            elif not modifier_text.strip():
                st.warning("Vui lòng nhập mô tả sự thay đổi mong muốn!")
            else:
                with st.spinner("Đang phân tích ý định và tìm kiếm trong kho dữ liệu..."):
                    with torch.no_grad():
                        # Phase 1: Reference image feature projection
                        inputs = vision_processor(images=[ref_image], return_tensors="pt").to(device)
                        outputs = vision_model.get_image_features(**inputs)
                        
                        # API structure verification layer
                        if hasattr(outputs, 'image_embeds'):
                            ref_vec = outputs.image_embeds
                        elif isinstance(outputs, torch.Tensor):
                            ref_vec = outputs
                        else:
                            ref_vec = vision_model.vision_model(**inputs).pooler_output
                            ref_vec = vision_model.visual_projection(ref_vec)
                            
                        ref_vec = ref_vec / ref_vec.norm(p=2, dim=-1, keepdim=True)
                        
                        # Phase 2: Natural language feature embedding
                        text_vec = text_model.encode([modifier_text], convert_to_tensor=True, device=device)
                        text_vec = F.normalize(text_vec, p=2, dim=-1)
                        
                        # Phase 3: Gated non-linear fusion propagation
                        query_vec = combiner(ref_vec, text_vec)
                        
                        # Phase 4: Cosine similarity computation across the visual corpus manifold
                        similarities = torch.matmul(img_tensors, query_vec.squeeze())
                        scores, top_indices = torch.topk(similarities, Config.TOP_K_RETRIEVAL)
                    
                    # Material UI top-k spatial rendering
                    res_cols = st.columns(3, gap="medium")
                    for i, idx in enumerate(top_indices.cpu().numpy()):
                        result_id = index_to_id[idx]
                        score = scores[i].item()
                        try:
                            res_img_path = os.path.join(Config.IMAGE_DIR, f"{result_id}.jpg")
                            with res_cols[i % 3]:
                                st.image(Image.open(res_img_path), width="stretch")
                                # Render the quantitative similarity matrix metrics
                                st.markdown(
                                    f"<div class='result-caption'>"
                                    f"ID: <b>{result_id}</b> <br>"
                                    f"<span class='score-badge'>Độ khớp: {score:.2f}</span>"
                                    f"</div>", 
                                    unsafe_allow_html=True
                                )
                        except Exception:
                            st.error(f"Lỗi đọc ảnh: {result_id}")
        else:
            # Empty state spatial placeholder
            st.info("💡 Hãy tải ảnh lên và nhập mô tả để khám phá các sản phẩm thời trang.")

if __name__ == "__main__":
    main()
