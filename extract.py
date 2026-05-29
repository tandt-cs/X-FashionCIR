import os, json, torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from config import Config

def extract_features():
    Config.setup_directories()
    # Tự động nhận diện GPU (nếu torch.cuda.is_available() là True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Sử dụng thiết bị: {device.upper()}")
    
    model = CLIPModel.from_pretrained(Config.VISION_MODEL_ID).to(device).eval()
    processor = CLIPProcessor.from_pretrained(Config.VISION_MODEL_ID)
    
    if not os.path.exists(Config.IMAGE_DIR): 
        print(f"Lỗi: Không tìm thấy thư mục {Config.IMAGE_DIR}")
        return
        
    files = [f for f in os.listdir(Config.IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Tìm thấy {len(files)} ảnh.")
    
    all_embeddings, id_to_index = [], {}
    
    with torch.no_grad():
        for i in tqdm(range(0, len(files), Config.BATCH_SIZE), desc="Trích xuất Vector"):
            batch = files[i : i + Config.BATCH_SIZE]
            imgs, valid = [], []
            for f in batch:
                try:
                    imgs.append(Image.open(os.path.join(Config.IMAGE_DIR, f)).convert("RGB"))
                    valid.append(f)
                except: pass
            
            if not imgs: continue
            
            inputs = processor(images=imgs, return_tensors="pt").to(device)
            
            # Cập nhật API lấy đặc trưng hình ảnh cho bản transformers mới
            outputs = model.get_image_features(**inputs)
            
            # Kiểm tra xem output có phải tensor không, nếu không lấy thuộc tính image_embeds
            if hasattr(outputs, 'image_embeds'):
                feats = outputs.image_embeds
            elif isinstance(outputs, torch.Tensor):
                 feats = outputs
            else:
                 # Trường hợp trả về BaseModelOutputWithPooling
                 feats = model.vision_model(**inputs).pooler_output
                 # Chiếu (project) vector về đúng không gian của CLIP
                 feats = model.visual_projection(feats)
            
            # Chuẩn hóa (L2 Normalization)
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            feats_np = feats.cpu().numpy()
            
            for j, f in enumerate(valid):
                id_to_index[os.path.splitext(f)[0]] = len(all_embeddings)
                all_embeddings.append(feats_np[j])

    np.save(Config.OUTPUT_EMBEDDINGS, np.array(all_embeddings, dtype=np.float32))
    with open(Config.OUTPUT_INDEX_MAP, 'w') as f:
        json.dump(id_to_index, f)
    print("Trích xuất hoàn tất!")

if __name__ == "__main__":
    extract_features()