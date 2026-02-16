
import os
import sys
import argparse
import time
from datetime import datetime
from dotenv import load_dotenv

# Import our modular engines
from ocr_engine import evaluate_with_gpt, locate_and_annotate

# Load environment variables
load_dotenv()

def process_scanner_batch(input_dir: str, output_dir: str):
    """
    Process all images in the input directory:
    1. Send to GPT for grading.
    2. Annotate using OCR.
    3. Save to output directory.
    """
    
    # Validate directories
    if not os.path.exists(input_dir):
        print(f"[ERROR] Input directory not found: {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of images
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
    
    total_files = len(files)
    print(f"\n[Scanner Processor] Found {total_files} images in '{input_dir}'")
    print(f"[Scanner Processor] Output will be saved to '{output_dir}'\n")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR] OPENAI_API_KEY not set in .env file.")
        return

    processed_count = 0
    
    for i, filename in enumerate(files):
        image_path = os.path.join(input_dir, filename)
        output_filename = f"GRADED_{filename}"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"[{i+1}/{total_files}] Processing: {filename}...")
        
        try:
            # 1. Evaluate with GPT-4o
            print("  > Sending to GPT-4o for grading...")
            start_time = time.time()
            evaluation = evaluate_with_gpt(
                image_path=image_path,
                api_key=api_key,
                preprocess=True # Ensure we preprocess scanned images if needed
            )
            gpt_time = time.time() - start_time
            print(f"  > GPT Response received in {gpt_time:.2f}s")
            
            if not evaluation:
                print("  > [WARNING] Empty response from GPT. Skipping.")
                continue

            # 2. Annotate with PaddleOCR
            print("  > Annotating image...")
            annotated_path = locate_and_annotate(
                image_path=image_path,
                gpt_response=evaluation,
                output_path=output_path
            )
            
            if annotated_path:
                print(f"  > ✓ Saved: {output_path}")
                processed_count += 1
            else:
                print(f"  > [ERROR] Failed to save annotated image.")
                
        except Exception as e:
            print(f"  > [ERROR] Failed to process {filename}: {e}")
            
        print("-" * 50)

    print(f"\n[Scanner Processor] Batch complete. {processed_count}/{total_files} processed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process scanned worksheets for grading.")
    
    parser.add_argument("input", help="Path to the folder containing scanned images")
    parser.add_argument("--output", default="scanned_graded", help="Path to save graded images (default: scanned_graded)")
    
    args = parser.parse_args()
    
    process_scanner_batch(args.input, args.output)
