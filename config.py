import os
import ssl

# ==========================================
# STANDARD SSL CONFIGURATION (ENTERPRISE SECURITY)
# ==========================================
try:
    import certifi
    # Default configuration utilizing the secure certificate bundle from the certifi package
    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
    os.environ['CURL_CA_BUNDLE'] = certifi.where()
except ImportError:
    pass

# ==========================================
# DIRECTORY AND HYPERPARAMETER CONFIGURATION
# ==========================================
class Config:
    BASE_DIR = "data"
    IMAGE_DIR = os.path.join(BASE_DIR, "images")
    CAPTIONS_DIR = os.path.join(BASE_DIR, "captions")
    
    # Storage directory for manually downloaded models (Offline Mode)
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    
    # Data file paths
    INPUT_JSON_PATH = os.path.join(CAPTIONS_DIR, "cap.dress.train.json")
    OUTPUT_JSON_PATH = os.path.join(CAPTIONS_DIR, "cap.dress.train.vn.json")

    # Vector embedding paths
    OUTPUT_EMBEDDINGS = os.path.join(BASE_DIR, "image_embeddings.npy")
    OUTPUT_INDEX_MAP = os.path.join(BASE_DIR, "image_id_to_index.json")

    # Models: Direct path configurations to the local storage directory
    VISION_MODEL_ID = os.path.join(MODELS_DIR, "clip-vit-base-patch32")
    TEXT_MODEL_ID = os.path.join(MODELS_DIR, "clip-ViT-B-32-multilingual-v1")

    # Global Hyperparameters
    BATCH_SIZE = 64
    ALPHA = 2.0
    TOP_K_RETRIEVAL = 6

    @classmethod
    def setup_directories(cls):
        os.makedirs(cls.IMAGE_DIR, exist_ok=True)
        os.makedirs(cls.CAPTIONS_DIR, exist_ok=True)
        os.makedirs(cls.MODELS_DIR, exist_ok=True)
