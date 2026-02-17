
import os
import sys
import time
import cv2
import concurrent.futures
from datetime import datetime
from dotenv import load_dotenv

# Import our modular components
from camera_system import capture_dual_cameras, init_cameras, release_cameras
from ocr_engine import evaluate_with_gpt, locate_and_annotate
from hardware_interface import ArduinoController

# Load environment variables
load_dotenv()

# Configuration
COM_PORT = "COM4"
BAUD_RATE = 9600
API_KEY = os.getenv("OPENAI_API_KEY")

def process_single_image(camera_name, paths):
    """
    Process a single image:
    1. GPT Evaluation
    2. OCR Annotation
    3. Saving result
    """
    if not paths:
        return f"[{camera_name}] No image captured."
        
    print(f"[{camera_name}] Starting processing...")
    
    gpt_image_path = paths.get("gpt_compressed")
    ocr_image_path = paths.get("cropped_gray") # Use rotated gray image
    
    if not gpt_image_path or not ocr_image_path:
        return f"[{camera_name}] Error: Missing paths."

    try:
        # 1. GPT Evaluation
        eval_result = evaluate_with_gpt(
            image_path=gpt_image_path, # Provide path, let engine handle reading
            api_key=API_KEY,
            preprocess=False # Image is already optimized by camera_system
        )
        
        if not eval_result:
            return f"[{camera_name}] GPT returned empty result."

        # 2. OCR Annotation
        output_path = ocr_image_path.replace(".png", "_GRADED.png")
        locate_and_annotate(
            image_path=ocr_image_path,
            gpt_response=eval_result,
            output_path=output_path
        )
        
        return f"[{camera_name}] ✓ Processed successfully. Saved to: {output_path}"
        
    except Exception as e:
        return f"[{camera_name}] Processing Error: {e}"


def main_loop():
    print("="*60)
    print("AUTOMATED GRADING SYSTEM with HARDWARE INTERGRATION")
    print("="*60)
    
    # 1. Initialize Hardware
    try:
        arduino = ArduinoController(port=COM_PORT, baud_rate=BAUD_RATE)
    except Exception as e:
        print("[System] FATAL: Hardware connection failed. Exiting.")
        return

    # 2. Initialize Cameras
    cam1, cam2 = init_cameras(0, 1)
    if not cam1 or not cam2:
        print("[System] FATAL: Camera initialization failed.")
        arduino.close()
        return

    print("\n[System] Starting Main Loop...")
    print("[System] Press Ctrl+C to stop.\n")
    
    cycle_count = 0
    
    try:
        while True:
            cycle_count += 1
            print(f"\n>>> STARTING CYCLE {cycle_count}")
            
            # --- STEP 1: CAPTURE ---
            print("[System] Capturing images...")
            # Capture from both cameras (Sequential capture is fast enough)
            # We pass 'None' for validation_callback to skip blocking internal logic
            # We just want the paths
            capture_data_cam1 = capture_dual_cameras(cam1, "cam1", "captured_copies")
            capture_data_cam2 = capture_dual_cameras(cam2, "cam2", "captured_copies")
            
            # --- STEP 2: PARALLEL PROCESSING ---
            print("[System] Processing images in PARALLEL...")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                # Submit both tasks
                future_cam1 = executor.submit(process_single_image, "cam1", capture_data_cam1)
                future_cam2 = executor.submit(process_single_image, "cam2", capture_data_cam2)
                
                # Wait for both to complete
                result_cam1 = future_cam1.result()
                result_cam2 = future_cam2.result()
                
            print(result_cam1)
            print(result_cam2)
            
            # --- STEP 3: HARDWARE FLIP ---
            print("[System] Cycle Complete. Sending FLIP signal...")
            arduino.send_flip_signal()
            
            # --- STEP 4: WAIT FOR CONFIRMATION ---
            print("[System] Waiting for hardware to finish...")
            if not arduino.wait_for_capture_signal():
                print("[System] Hardware timeout or error. Retrying wait...")
                # Optional: Decide whether to crash or retry
                # For now, let's wait again or break
                # break 
            
            # If we get here, hardware said "capture", so we loop back immediately
            print("[System] Hardware ready. Continuing to next cycle.")
            
    except KeyboardInterrupt:
        print("\n[System] Stopping...")
        
    finally:
        arduino.close()
        release_cameras(cam1, cam2)
        print("[System] Shutdown complete.")

if __name__ == "__main__":
    main_loop()
