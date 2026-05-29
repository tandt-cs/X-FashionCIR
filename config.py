import os
import ssl

# ==========================================
# CẤU HÌNH SSL CHUẨN (AN TOÀN DOANH NGHIỆP)
# ==========================================
# Nếu IT công ty cung cấp file chứng chỉ (Root CA), hãy bỏ comment và sửa đường dẫn này:
# os.environ['REQUESTS_CA_BUNDLE'] = "C:/path/to/company/rootCA.crt"
# os.environ['CURL_CA_BUNDLE'] = "C:/path/to/company/rootCA.crt"

try:
    import certifi
    # Thiết lập mặc định sử dụng chứng chỉ an toàn của gói certifi
    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
    os.environ['CURL_CA_BUNDLE'] = certifi.where()
except ImportError:
    pass

# ==========================================
# CẤU HÌNH THƯ MỤC VÀ THAM SỐ
# ==========================================
class Config:
    BASE_DIR = "data"
    IMAGE_DIR = os.path.join(BASE_DIR, "images")
    CAPTIONS_DIR = os.path.join(BASE_DIR, "captions")
    
    # Thư mục lưu trữ mô hình tải thủ công (Offline Mode)
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    
    # Files dữ liệu
    INPUT_JSON_PATH = os.path.join(CAPTIONS_DIR, "cap.dress.train.json")
    OUTPUT_JSON_PATH = os.path.join(CAPTIONS_DIR, "cap.dress.train.vn.json")

    # Files Vector
    OUTPUT_EMBEDDINGS = os.path.join(BASE_DIR, "image_embeddings.npy")
    OUTPUT_INDEX_MAP = os.path.join(BASE_DIR, "image_id_to_index.json")

    # Models: Cấu hình trỏ trực tiếp vào thư mục trên ổ cứng nội bộ
    VISION_MODEL_ID = os.path.join(MODELS_DIR, "clip-vit-base-patch32")
    TEXT_MODEL_ID = os.path.join(MODELS_DIR, "clip-ViT-B-32-multilingual-v1")

    # Hyperparameters
    BATCH_SIZE = 64
    ALPHA = 2.0
    TOP_K_RETRIEVAL = 6

    @classmethod
    def setup_directories(cls):
        os.makedirs(cls.IMAGE_DIR, exist_ok=True)
        os.makedirs(cls.CAPTIONS_DIR, exist_ok=True)
        os.makedirs(cls.MODELS_DIR, exist_ok=True)