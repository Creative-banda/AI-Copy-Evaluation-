"""
Main Application - Document Capture & OCR Grading System

This is the main entry point that combines:
    - Dual camera document capture (camera_system.py)
    - OCR text detection with preprocessing (ocr_engine.py)
    - Worksheet grading and annotation

Usage:
    1. Camera Mode (capture documents):
       python main.py camera
    
    2. OCR Mode (find text in images):
       python main.py ocr <image_path> <search_text> [--engine paddle|tesseract]
    
    3. Grade Mode (evaluate worksheet):
       python main.py grade <image_path> --qa "Question 1" "Answer 1" --qa "Question 2" "Answer 2"
    
    4. Batch OCR (process all images in folder):
       python main.py batch <images_folder> <search_text> [--engine paddle|tesseract]
"""

import sys
import os
import argparse
from pathlib import Path

# Import our modules
try:
    from camera_system import start_camera_system
    from ocr_engine import (
        find_text_with_paddle,
        find_text_with_tesseract,
        get_all_text_paddle,
        preprocess_image,
        evaluate_with_gpt,
        draw_boxes_on_image,
        locate_and_annotate,
        extract_text_with_layout,
        WordBox
    )
    # NEW: Import Handwriting System
    from handwriting_system import handwriting
    from dotenv import load_dotenv
    load_dotenv()
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure camera_system.py and ocr_engine.py are in the same directory")
    sys.exit(1)

try:
    from PIL import Image, ImageDraw, ImageFont
    import cv2
except ImportError:
    print("Missing dependencies. Install with:")
    print("  pip install opencv-python pillow paddleocr pytesseract")
    sys.exit(1)


# ============================================================================
# CAMERA MODE
# ============================================================================


def run_camera_mode(args):
    """Run dual camera document capture system with HARDWARE AUTOMATION"""
    import threading
    
    print("Starting process with PARALLEL PROCESSING...")
    print("CYCLE: Parallel Capture & Grade -> Wait for ALL -> Flip -> Repeat")
    print()
    
    # Initialize Arduino
    try:
        from hardware_interface import ArduinoController
        arduino = ArduinoController(port="COM3", baud_rate=115200) # Updated to COM3
    except Exception as e:
        print(f"[System] WARNING: Hardware connection failed: {e}")
        print("[System] Running in SIMULATION MODE (No Hardware)")
        arduino = None

    # Shared state for synchronization
    processing_lock = threading.Lock()
    cycle_lock = threading.Lock()
    
    # Trackers
    camera_handlers = {}  # Store handler instances to control pause/resume
    processed_cameras = set()
    
    def grade_and_sync(camera_name, data):
        """Thread function: Grade image -> Mark done -> Check if all done -> Flip"""
        try:
            # 1. GRADE INDEPENDENTLY
            process_single_image_grading(camera_name, data)
            
            # 2. UPDATE STATE
            ready_to_flip = False
            with processing_lock:
                processed_cameras.add(camera_name)
                # Check if we have both cameras (cam1 and cam2)
                # We assume 2 cameras are active
                if len(processed_cameras) >= 2:
                    ready_to_flip = True
                    
            # 3. TRIGGER FLIP (Only the last finishing thread does this)
            if ready_to_flip:
                with cycle_lock:
                    print(f"\n[System] All cameras ({processed_cameras}) processed. Syncing...")
                    
                    # Send Flip
                    print("[System] Sending FLIP signal... (SKIPPED FOR TESTING)")
                    # if arduino: arduino.send_flip_signal()
                    
                    # Wait for Hardware
                    print("[System] Waiting for hardware... (SKIPPED FOR TESTING)")
                    import time
                    time.sleep(2) # Simulate flip delay
                    
                    # if arduino and not arduino.wait_for_capture_signal():
                    #     print("[System] WARNING: Hardware timeout.")
                    # else:
                    #     print("[System] Hardware ready.")
                        
                    # RESET CYCLE AND RESUME CAMERAS
                    print("[System] Resuming cameras...")
                    for name, handler in camera_handlers.items():
                        if handler:
                            handler.resume()
                            
                    with processing_lock:
                        camera_handlers.clear()
                        processed_cameras.clear()
                    print("-" * 50)
                    print("Ready for next cycle...")
                    
        except Exception as e:
            print(f"[{camera_name}] Thread Error: {e}")

    # Callback function to handle captured images
    def on_capture_callback(data):
        camera_name = data.get("camera_name", "Unknown")
        handler = data.get("handler")
        
        with processing_lock:
            # IGNORE duplicate captures (safety check)
            if camera_name in camera_handlers:
                return
            
            # Store handler and PAUSE immediately
            camera_handlers[camera_name] = handler
            if handler:
                handler.pause()
                print(f"[{camera_name}] Paused detection.")

        print(f"[{camera_name}] Captured! Starting Grading Thread...")
        
        # Start grading in a separate thread
        t = threading.Thread(target=grade_and_check_wrapper, args=(camera_name, data))
        t.start()

    def grade_and_check_wrapper(camera_name, data):
         # Wrapper to call the internal logic
         grade_and_sync(camera_name, data)

    def process_single_image_grading(camera_name, data):
        """Helper to grade a single image"""
        paths = data.get("paths", {})
        gpt_image_path = paths.get("gpt_compressed")
        ocr_image_path = paths.get("cropped_gray")
        
        if not gpt_image_path or not ocr_image_path:
            print(f"[{camera_name}] Error: Missing paths")
            return

        print(f"[{camera_name}] Grading in progress...")
        try:
            # Evaluate
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key: 
                print(f"[{camera_name}] No API Key")
                return

            evaluation = evaluate_with_gpt(
                image_path=gpt_image_path,
                api_key=api_key,
                preprocess=False
            )
            
            # Annotate
            # Annotate
            output_path = ocr_image_path.replace('.png', '_GRADED.png')
            graded_path, annotations = locate_and_annotate(
                image_path=ocr_image_path,
                gpt_response=evaluation,
                output_path=output_path
            )
            
            # --- Handwriting Generation ---
            if annotations:
                print(f"[{camera_name}] Generating Handwriting G-Code...")
                try:
                    # Get image dimensions for correct scaling
                    # We can use PIL or cv2 since we have dependencies
                    # ocr_image_path is loaded in locate_and_annotate but not returned. Let's load just dims.
                    try:
                        with Image.open(ocr_image_path) as img_check:
                            width, height = img_check.size
                    except:
                        # Fallback if image load fails
                        width, height = 3840, 2160 # Default 4K

                    # Fix filename collision: include camera_name and timestamp
                    # ocr_image_path is like ".../captured_copies/cam1_2024.../original_gray.png"
                    # We can just append .svg to the full path to keep it in the folder
                    hw_filename = f"{camera_name}_{os.path.basename(ocr_image_path).replace('.png', '.svg')}"
                    
                    # Calculate simple score
                    correct_count = sum(1 for a in annotations if a['type'] == 'correct')
                    total_count = len(annotations)
                    score_text = f"{correct_count}/{total_count}"
                    
                    # Generate SVG & G-Code
                    # Note: generate_svg saves to self.output_dir which defaults to "handwriting_output"
                    # We might want to save it NEXT TO the image instead.
                    # But HandwritingSystem is init with output_dir="handwriting_output".
                    # Let's let it save there but with unique name.
                    
                    svg_path = handwriting.generate_svg(
                        hw_filename, 
                        annotations, 
                        score_text,
                        source_width=width,
                        source_height=height
                    )
                    gcode_path = handwriting.convert_to_gcode(svg_path)
                    
                    if gcode_path:
                         print(f"[{camera_name}] ✓ G-Code ready: {gcode_path}")
                except Exception as hw_error:
                    print(f"[{camera_name}] Handwriting Gen Error: {hw_error}")
            # ------------------------------
            print(f"[{camera_name}] ✓ Grading Complete")
            
        except Exception as e:
            print(f"[{camera_name}] Grading Error: {e}")

    try:
        start_camera_system(
            camera1_id=args.camera1,
            camera2_id=args.camera2,
            output_folder=args.output,
            on_capture=on_capture_callback
        )
    finally:
        if arduino: arduino.close()


# ============================================================================
# OCR MODE
# ============================================================================

def run_ocr_mode(args):
    """Find text in image using OCR"""
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        return 1
    
    print("=" * 70)
    print("OCR TEXT DETECTION")
    print("=" * 70)
    print(f"Image: {args.image}")
    print(f"Search: '{args.search}'")
    print(f"Engine: {args.engine.upper()}")
    print(f"Preprocessing: {'Enabled' if not args.no_preprocess else 'Disabled'}")
    print("=" * 70)
    
    # Find text
    if args.engine == "paddle":
        print("\nUsing PaddleOCR...")
        boxes = find_text_with_paddle(
            args.image,
            args.search,
            preprocess=not args.no_preprocess
        )
    else:  # tesseract
        print("\nUsing Tesseract...")
        boxes = find_text_with_tesseract(
            args.image,
            args.search,
            preprocess=not args.no_preprocess
        )
    
    # Display results
    if boxes:
        print(f"\n✓ Found {len(boxes)} match(es):")
        for i, box in enumerate(boxes, 1):
            conf_str = f"confidence={box.confidence:.2f}" if box.confidence > 0 else ""
            print(f"  {i}. '{box.text}' at ({box.left}, {box.top}, {box.right}, {box.bottom}) {conf_str}")
        
        # Save annotated image
        output_path = args.image.replace('.', '_annotated.')
        draw_boxes_on_image(args.image, boxes, output_path, color=(0, 255, 0), thickness=2)
        print(f"\n✓ Annotated image saved: {output_path}")
    else:
        print(f"\n✗ Text '{args.search}' not found in image")
    
    return 0


# ============================================================================
# BATCH OCR MODE
# ============================================================================

def run_batch_mode(args):
    """Process all images in a folder"""
    folder = Path(args.folder)
    
    if not folder.exists():
        print(f"Error: Folder not found: {folder}")
        return 1
    
    # Find all images
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    images = [f for f in folder.iterdir() if f.suffix.lower() in image_extensions]
    
    if not images:
        print(f"No images found in {folder}")
        return 1
    
    print("=" * 70)
    print("BATCH OCR PROCESSING")
    print("=" * 70)
    print(f"Folder: {folder}")
    print(f"Images: {len(images)}")
    print(f"Search: '{args.search}'")
    print(f"Engine: {args.engine.upper()}")
    print("=" * 70)
    
    results = []
    
    for img_path in images:
        print(f"\nProcessing: {img_path.name}")
        
        try:
            if args.engine == "paddle":
                boxes = find_text_with_paddle(str(img_path), args.search, preprocess=True)
            else:
                boxes = find_text_with_tesseract(str(img_path), args.search, preprocess=True)
            
            if boxes:
                print(f"  ✓ Found {len(boxes)} match(es)")
                results.append((img_path.name, True, len(boxes)))
                
                # Save annotated
                output_path = str(img_path).replace(img_path.suffix, f'_found{img_path.suffix}')
                draw_boxes_on_image(str(img_path), boxes, output_path)
            else:
                print(f"  ✗ Not found")
                results.append((img_path.name, False, 0))
        
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append((img_path.name, False, 0))
    
    # Summary
    print("\n" + "=" * 70)
    print("BATCH SUMMARY")
    print("=" * 70)
    found_count = sum(1 for _, found, _ in results if found)
    print(f"Total images: {len(images)}")
    print(f"Found in: {found_count}")
    print(f"Not found: {len(images) - found_count}")
    print("=" * 70)
    
    return 0


# ============================================================================
# GRADE MODE
# ============================================================================

def run_grade_mode(args):
    """Grade worksheet with OCR + GPT Vision"""
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        return 1
    
    if not args.qa:
        print("Error: No question-answer pairs provided")
        print("Use: --qa \"Question\" \"Answer\" (can use multiple times)")
        return 1
    
    print("=" * 70)
    print("WORKSHEET GRADING")
    print("=" * 70)
    print(f"Image: {args.image}")
    print(f"Questions: {len(args.qa)}")
    print(f"Engine: {args.engine.upper()}")
    print("=" * 70)
    
    # Try OCR first
    print("\n[1/2] OCR-based grading...")
    
    img = Image.open(args.image)
    draw = ImageDraw.Draw(img)
    
    correct_count = 0
    found_with_ocr = 0
    
    for i, (question, answer) in enumerate(args.qa):
        print(f"\nQ{i+1}: {question}")
        print(f"Expected: {answer}")
        
        # Search with OCR
        if args.engine == "paddle":
            boxes = find_text_with_paddle(args.image, answer, preprocess=True)
        else:
            boxes = find_text_with_tesseract(args.image, answer, preprocess=True)
        
        if boxes:
            found_with_ocr += 1
            print(f"  ✓ Found at ({boxes[0].left}, {boxes[0].top})")
            
            # Draw green box
            for box in boxes:
                draw.rectangle(
                    [(box.left, box.top), (box.right, box.bottom)],
                    outline="green",
                    width=3
                )
                draw.text(
                    (box.left, box.top - 20),
                    "✓ Correct",
                    fill="green"
                )
            correct_count += 1
        else:
            print(f"  ✗ Not found with OCR")
    
    # Try GPT Vision as fallback
    if found_with_ocr < len(args.qa) and args.api_key:
        print(f"\n[2/2] GPT Vision fallback for remaining {len(args.qa) - found_with_ocr} questions...")
        
        try:
            result = evaluate_with_gpt(
                args.image,
                args.qa,
                api_key=args.api_key,
                preprocess=True
            )
            
            # Draw GPT results
            for item in result.get("results", []):
                if "bbox" in item and item["bbox"]:
                    left, top, right, bottom = item["bbox"]
                    color = "green" if item["correct"] else "red"
                    label = "✓ Correct" if item["correct"] else "✗ Wrong"
                    
                    draw.rectangle(
                        [(left, top), (right, bottom)],
                        outline=color,
                        width=3
                    )
                    draw.text(
                        (left, top - 20),
                        label,
                        fill=color
                    )
            
            correct_count = result.get("total_correct", correct_count)
            print(f"  ✓ GPT found {result.get('total_correct', 0)} correct answers")
        
        except Exception as e:
            print(f"  ✗ GPT Vision failed: {e}")
    
    # Add score
    total = len(args.qa)
    score = (correct_count / total * 100) if total > 0 else 0
    
    draw.text(
        (10, 10),
        f"Score: {correct_count}/{total} ({score:.1f}%)",
        fill="blue",
        font=None
    )
    
    # Save
    img.save(args.output)
    
    print("\n" + "=" * 70)
    print("GRADING COMPLETE")
    print("=" * 70)
    print(f"Score: {correct_count}/{total} ({score:.1f}%)")
    print(f"Saved to: {args.output}")
    print("=" * 70)
    
    return 0


# ============================================================================
# PREPROCESS MODE
# ============================================================================

def run_preprocess_mode(args):
    """Preprocess image and save"""
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        return 1
    
    print("=" * 70)
    print("IMAGE PREPROCESSING")
    print("=" * 70)
    print(f"Input: {args.image}")
    print(f"Method: {args.method.upper()}")
    print(f"Output: {args.output}")
    print("=" * 70)
    
    preprocessed = preprocess_image(
        args.image,
        method=args.method,
        save_path=args.output
    )
    
    print(f"\n✓ Preprocessed image saved: {args.output}")
    print(f"  Shape: {preprocessed.shape}")
    print(f"  Method: {args.method}")
    
    return 0


def run_extract_mode(args):
    """Run text extraction mode"""
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        return 1
        
    print(f"Extracting text from: {args.image}")
    print("=" * 70)
    
    try:
        text = extract_text_with_layout(args.image, preprocess=True)
        print(text)
    except Exception as e:
        print(f"Error during extraction: {e}")
        return 1
        
    print("=" * 70)
    return 0


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Document Capture & OCR Grading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start camera capture system
  python main.py camera
  
  # Find text in image with PaddleOCR
  python main.py ocr images/worksheet.png "Question 1" --engine paddle
  
  # Grade worksheet
  python main.py grade images/test.png --qa "Q1" "Answer1" --qa "Q2" "Answer2"
  
  # Batch process folder
  python main.py batch images/ "student name" --engine tesseract
  
  # Preprocess image
  python main.py preprocess images/scan.png --method aggressive
        """
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Camera mode
    camera_parser = subparsers.add_parser('camera', help='Start dual camera capture')
    camera_parser.add_argument('--camera1', type=int, default=0, help='First camera ID')
    camera_parser.add_argument('--camera2', type=int, default=1, help='Second camera ID')
    camera_parser.add_argument('--output', default='captured_copies', help='Output folder')
    
    # Extract mode
    extract_parser = subparsers.add_parser('extract', help='Extract text from image preserving layout')
    extract_parser.add_argument('image', help='Path to input image')
    
    # OCR mode
    ocr_parser = subparsers.add_parser('ocr', help='Find text in image')
    ocr_parser.add_argument('image', help='Image file path')
    ocr_parser.add_argument('search', help='Text to search for')
    ocr_parser.add_argument('--engine', choices=['paddle', 'tesseract'], 
                           default='paddle', help='OCR engine')
    ocr_parser.add_argument('--no-preprocess', action='store_true',
                           help='Disable preprocessing')
    
    # Batch mode
    batch_parser = subparsers.add_parser('batch', help='Process folder of images')
    batch_parser.add_argument('folder', help='Folder containing images')
    batch_parser.add_argument('search', help='Text to search for')
    batch_parser.add_argument('--engine', choices=['paddle', 'tesseract'],
                             default='paddle', help='OCR engine')
    
    # Grade mode
    grade_parser = subparsers.add_parser('grade', help='Grade worksheet')
    grade_parser.add_argument('image', help='Worksheet image path')
    grade_parser.add_argument('--qa', action='append', nargs=2,
                             metavar=('QUESTION', 'ANSWER'),
                             help='Question-answer pair (use multiple times)')
    grade_parser.add_argument('--engine', choices=['paddle', 'tesseract'],
                             default='paddle', help='OCR engine')
    grade_parser.add_argument('--api-key', help='OpenAI API key for GPT fallback')
    grade_parser.add_argument('--output', default='graded.png', help='Output image')
    
    # Preprocess mode
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess image')
    preprocess_parser.add_argument('image', help='Input image path')
    preprocess_parser.add_argument('--method', choices=['light', 'aggressive'],
                                   default='light', help='Preprocessing method')
    preprocess_parser.add_argument('--output', help='Output image path')
    
    args = parser.parse_args()
    
    # Handle no mode - Default to camera mode
    if not args.mode:
        print("No mode specified, defaulting to CAMERA mode...")
        # Create dummy args for camera mode
        args.mode = 'camera'
        args.camera1 = 0
        args.camera2 = 1
        args.output = 'captured_copies'
        return run_camera_mode(args)
    
    # Set defaults
    if args.mode == 'preprocess' and not args.output:
        args.output = args.image.replace('.', '_preprocessed.')
    
    # Route to appropriate handler
    if args.mode == 'camera':
        return run_camera_mode(args)
    elif args.mode == 'extract':
        return run_extract_mode(args)
    elif args.mode == 'ocr':
        return run_ocr_mode(args)
    elif args.mode == 'batch':
        return run_batch_mode(args)
    elif args.mode == 'grade':
        return run_grade_mode(args)
    elif args.mode == 'preprocess':
        return run_preprocess_mode(args)
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
