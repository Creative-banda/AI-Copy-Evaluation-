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
from typing import List, Tuple

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
    """Run dual camera document capture system with AUTO GRADING"""
    print("Starting camera system with AUTO GRADING...")
    print("Captured documents will be automatically:")
    print("  1. Enhanced for OCR")
    print("  2. Analyzed by GPT-4o")
    print("  3. Annotated with Correct/Wrong")
    print()
    
    # Callback function to handle captured images
    def on_capture_callback(data):
        camera_name = data.get("camera_name", "Unknown")
        paths = data.get("paths", {})
        
        # 1. Use Compressed image for GPT (Save Tokens/Bandwidth)
        gpt_image_path = paths.get("gpt_compressed")
        
        # 2. Use Normal Cropped Grayscale image for OCR (Standard Quality)
        ocr_image_path = paths.get("cropped_gray")
        
        if not gpt_image_path or not ocr_image_path:
            print(f"[{camera_name}] Error: Missing required image paths")
            return
            
        print(f"\n[{camera_name}] Auto-Grading...")
        print(f"[{camera_name}] GPT Input: {os.path.basename(gpt_image_path)}")
        print(f"[{camera_name}] OCR Input: {os.path.basename(ocr_image_path)}")
        
        try:
            # 1. Evaluate with GPT (using compressed image)
            print(f"[{camera_name}] Sending to GPT Vision...")
            api_key = os.getenv("OPENAI_API_KEY")
            
            if not api_key:
                print(f"[{camera_name}] Error: OPENAI_API_KEY not found in .env")
                return

            evaluation = evaluate_with_gpt(
                image_path=gpt_image_path,
                api_key=api_key,
                preprocess=False  # Already processed
            )
            
            # 2. Annotate Image (using High Quality image)
            print(f"[{camera_name}] Locating questions and annotating...")
            
            # Save graded image in the SAME folder as the capture
            output_path = ocr_image_path.replace('.png', '_GRADED.png')
            
            annotated_path = locate_and_annotate(
                image_path=ocr_image_path,
                gpt_response=evaluation,
                output_path=output_path
            )
            
            print(f"[{camera_name}] ✓ GRADING COMPLETE!")
            print(f"[{camera_name}] Saved to: {annotated_path}")
            
            # Print summary
            total_q = len(evaluation)
            correct_q = sum(1 for v in evaluation.values() if v.get("isAnswerCorrect"))
            score = (correct_q / total_q * 100) if total_q > 0 else 0
            print(f"[{camera_name}] Score: {correct_q}/{total_q} ({score:.1f}%)")
            
        except Exception as e:
            print(f"[{camera_name}] Error during auto-grading: {e}")
            import traceback
            traceback.print_exc()

    start_camera_system(
        camera1_id=args.camera1,
        camera2_id=args.camera2,
        output_folder=args.output,
        on_capture=on_capture_callback
    )


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
