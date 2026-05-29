import os, json, time, logging
from tqdm import tqdm
from deep_translator import GoogleTranslator
from config import Config

# Cấu hình lưu log ra cả màn hình và file text để dễ theo dõi
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler("translation.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def translate_with_retry(translator, text, retries=3):
    """Hàm dịch an toàn: Trả về Tuple (văn_bản_đã_dịch, trạng_thái_thành_công)"""
    for attempt in range(retries):
        try:
            result = translator.translate(text)
            if result:
                return result, True
        except Exception as e:
            time.sleep(2) # Đợi 2s trước khi thử lại để tránh rate limit
            
    # Nếu thử 3 lần vẫn thất bại, giữ nguyên tiếng Anh và báo False
    logging.warning(f"Không thể dịch: '{text}'. Giữ nguyên bản gốc.")
    return text, False

def translate_dataset():
    Config.setup_directories()
    translator = GoogleTranslator(source='en', target='vi')
    captions_dir = Config.CAPTIONS_DIR
    
    if not os.path.exists(captions_dir):
        logging.error(f"Không tìm thấy thư mục: {captions_dir}")
        return
        
    all_files = os.listdir(captions_dir)
    # Lọc ra các file JSON gốc (tiếng Anh), bỏ qua các file đã dịch (.vn.json) và file lỗi (_failed.json)
    en_files = [f for f in all_files if f.startswith('cap.') and f.endswith('.json') and not f.endswith('.vn.json') and not f.endswith('_failed.json')]
    
    if not en_files:
        logging.info("Không tìm thấy file tiếng Anh nào cần dịch.")
        return

    logging.info(f"Phát hiện {len(en_files)} file cần dịch: {en_files}")
    
    for file in en_files:
        input_path = os.path.join(captions_dir, file)
        output_path = os.path.join(captions_dir, file.replace('.json', '.vn.json'))
        failed_log_path = output_path.replace('.json', '_failed.json')
        
        logging.info(f"\n=== ĐANG XỬ LÝ FILE: {file} ===")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        translated_data = []
        failed_records = []
        
        # Load lại dữ liệu đã dịch để chạy tiếp (Resume)
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                translated_data = json.load(f)
                
        # Load lại danh sách lỗi cũ (nếu có)
        if os.path.exists(failed_log_path):
            with open(failed_log_path, 'r', encoding='utf-8') as f:
                failed_records = json.load(f)
                
        processed = {item['candidate'] for item in translated_data}
        remaining = [item for item in raw_data if item['candidate'] not in processed]
        
        if not remaining:
            logging.info(f"File {file} đã được dịch hoàn tất từ trước. Chuyển sang file tiếp theo.")
            continue
            
        logging.info(f"Cần dịch: {len(remaining)} mục trong file {file}.")
        
        try:
            for idx, item in enumerate(tqdm(remaining, desc=f"Đang dịch {file}")):
                new_item = item.copy()
                captions_vn = []
                has_error = False
                
                # Xử lý từng câu và lấy trạng thái thành công/thất bại
                for cap in item['captions']:
                    translated_text, success = translate_with_retry(translator, cap)
                    captions_vn.append(translated_text)
                    if not success:
                        has_error = True
                
                new_item['captions_vn'] = captions_vn
                translated_data.append(new_item)
                
                # Nếu có ít nhất 1 câu bị lỗi, đưa vào danh sách failed
                if has_error:
                    failed_records.append({
                        "candidate": item["candidate"],
                        "original_captions": item["captions"],
                        "fallback_captions": captions_vn
                    })
                
                time.sleep(0.3) # Nghỉ để tránh bị Google block
                
                # Lưu tiến độ mỗi 50 câu
                if (idx + 1) % 50 == 0:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(translated_data, f, ensure_ascii=False, indent=4)
                    with open(failed_log_path, 'w', encoding='utf-8') as f:
                        json.dump(failed_records, f, ensure_ascii=False, indent=4)
        finally:
            # Luôn đảm bảo lưu data khi chương trình dừng hoặc qua file mới
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(translated_data, f, ensure_ascii=False, indent=4)
            with open(failed_log_path, 'w', encoding='utf-8') as f:
                json.dump(failed_records, f, ensure_ascii=False, indent=4)
                
            logging.info(f"Đã xử lý xong file {file}.")
            if failed_records:
                logging.warning(f"Có {len(failed_records)} lỗi trong file này. Xem chi tiết tại: {failed_log_path}")

    logging.info("\nĐÃ HOÀN TẤT DỊCH TOÀN BỘ CÁC BỘ DỮ LIỆU!")

if __name__ == "__main__":
    translate_dataset()