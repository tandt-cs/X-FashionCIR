import os, json, time, logging
from tqdm import tqdm
from deep_translator import GoogleTranslator
from config import Config

# Configure dual logging (Console and Text File) for robust execution tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler("translation.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def translate_with_retry(translator, text, retries=3):
    """Safely translates text with an exponential backoff strategy. Returns Tuple(translated_text, success_status)"""
    for attempt in range(retries):
        try:
            result = translator.translate(text)
            if result:
                return result, True
        except Exception as e:
            time.sleep(2) # Enforce a 2-second delay to mitigate API rate limiting
            
    # Retain the original English text if all retry attempts fail
    logging.warning(f"Translation failed for: '{text}'. Retaining the original caption.")
    return text, False

def translate_dataset():
    Config.setup_directories()
    translator = GoogleTranslator(source='en', target='vi')
    captions_dir = Config.CAPTIONS_DIR
    
    if not os.path.exists(captions_dir):
        logging.error(f"Directory not found: {captions_dir}")
        return
        
    all_files = os.listdir(captions_dir)
    # Filter for standard English JSON files, bypassing localized (.vn.json) and failed (_failed.json) artifacts
    en_files = [f for f in all_files if f.startswith('cap.') and f.endswith('.json') and not f.endswith('.vn.json') and not f.endswith('_failed.json')]
    
    if not en_files:
        logging.info("No English corpus files found requiring translation.")
        return

    logging.info(f"Detected {len(en_files)} files scheduled for localization: {en_files}")
    
    for file in en_files:
        input_path = os.path.join(captions_dir, file)
        output_path = os.path.join(captions_dir, file.replace('.json', '.vn.json'))
        failed_log_path = output_path.replace('.json', '_failed.json')
        
        logging.info(f"\n=== INITIATING PROCESSING PIPELINE FOR: {file} ===")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        translated_data = []
        failed_records = []
        
        # Load previously localized data to resume execution efficiently
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                translated_data = json.load(f)
                
        # Load the legacy error logs if they exist
        if os.path.exists(failed_log_path):
            with open(failed_log_path, 'r', encoding='utf-8') as f:
                failed_records = json.load(f)
                
        processed = {item['candidate'] for item in translated_data}
        remaining = [item for item in raw_data if item['candidate'] not in processed]
        
        if not remaining:
            logging.info(f"Corpus file {file} is completely localized. Proceeding to the next sequence.")
            continue
            
        logging.info(f"Translation queue: {len(remaining)} instances pending in {file}.")
        
        try:
            for idx, item in enumerate(tqdm(remaining, desc=f"Localizing {file}")):
                new_item = item.copy()
                captions_vn = []
                has_error = False
                
                # Iterate through individual annotations and monitor the success matrix
                for cap in item['captions']:
                    translated_text, success = translate_with_retry(translator, cap)
                    captions_vn.append(translated_text)
                    if not success:
                        has_error = True
                
                new_item['captions_vn'] = captions_vn
                translated_data.append(new_item)
                
                # Document the anomaly if any specific caption sequence fails
                if has_error:
                    failed_records.append({
                        "candidate": item["candidate"],
                        "original_captions": item["captions"],
                        "fallback_captions": captions_vn
                    })
                
                time.sleep(0.3) # Introduce micro-latency to bypass anti-bot mechanisms
                
                # Checkpoint preservation per 50 iterations
                if (idx + 1) % 50 == 0:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(translated_data, f, ensure_ascii=False, indent=4)
                    with open(failed_log_path, 'w', encoding='utf-8') as f:
                        json.dump(failed_records, f, ensure_ascii=False, indent=4)
        finally:
            # Guarantee data persistence upon process termination or file completion
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(translated_data, f, ensure_ascii=False, indent=4)
            with open(failed_log_path, 'w', encoding='utf-8') as f:
                json.dump(failed_records, f, ensure_ascii=False, indent=4)
                
            logging.info(f"Processing concluded for corpus: {file}.")
            if failed_records:
                logging.warning(f"Recorded {len(failed_records)} translation anomalies. Detailed logs available at: {failed_log_path}")

    logging.info("\nENTIRE DATASET LOCALIZATION PROTOCOL COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    translate_dataset()
