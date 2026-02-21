"""
OCR Engine Module - PaddleOCR & Tesseract with Image Preprocessing

This module provides high-accuracy OCR functions with automatic preprocessing.
Export functions:
    - find_text_with_paddle() - PaddleOCR text detection
    - find_text_with_tesseract() - Tesseract text detection
    - preprocess_image() - Image preprocessing for better OCR
    - evaluate_with_gpt() - GPT Vision API evaluation
"""

import os
import base64
import json
import re
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class WordBox:
    """Bounding box for detected text"""
    text: str
    left: int
    top: int
    width: int
    height: int
    confidence: float = 0.0

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height


# ============================================================================
# IMAGE PREPROCESSING FUNCTIONS
# ============================================================================

def preprocess_image(
    image: Union[str, np.ndarray],
    method: str = "light",
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Preprocess image for optimal OCR accuracy
    
    Args:
        image: Path to image or numpy array
        method: "light" (grayscale+denoise) or "aggressive" (all filters)
        save_path: Optional path to save preprocessed image
        
    Returns:
        Preprocessed grayscale image
    """
    # Load image
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise ValueError(f"Cannot load image: {image}")
    else:
        img = image.copy()
    
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Apply denoising
    gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    if method == "aggressive":
        # Apply CLAHE contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Apply adaptive thresholding
        gray = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
    
    # Save if path provided
    if save_path:
        cv2.imwrite(save_path, gray)
    
    return gray


# ============================================================================
# PADDLEOCR ENGINE
# ============================================================================

class _PaddleOCREngine:
    """Internal PaddleOCR engine wrapper"""
    
    def __init__(self, lang='en', show_log=False):
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            raise ImportError(
                "PaddleOCR not installed. Install with:\n"
                "  pip install paddleocr==2.9.1"
            )
        self.ocr = PaddleOCR(lang=lang, show_log=show_log)
    
    def detect(self, image_path: str) -> List[Tuple[List, Tuple[str, float]]]:
        """Run OCR and return results"""
        result = self.ocr.ocr(image_path, cls=False)
        return result[0] if result and result[0] else []


# Global PaddleOCR instance (lazy initialization)
_paddle_engine = None

def _get_paddle_engine():
    """Get or create PaddleOCR engine instance"""
    global _paddle_engine
    if _paddle_engine is None:
        _paddle_engine = _PaddleOCREngine(lang='en', show_log=False)
    return _paddle_engine


def find_text_with_paddle(
    image_path: str,
    search_text: str,
    preprocess: bool = True,
    min_confidence: float = 0.5
) -> List[WordBox]:
    """
    Find text using PaddleOCR with optional preprocessing
    
    Args:
        image_path: Path to image file
        search_text: Text to search for (case-insensitive)
        preprocess: Apply preprocessing for better accuracy
        min_confidence: Minimum confidence threshold (0.0 to 1.0)
        
    Returns:
        List of WordBox objects with detected text regions
    """
    # Preprocess if enabled
    if preprocess:
        preprocessed = preprocess_image(image_path, method="light")
        temp_path = image_path.replace('.', '_paddle_temp.')
        cv2.imwrite(temp_path, preprocessed)
        process_path = temp_path
    else:
        process_path = image_path
    
    try:
        # Run OCR
        engine = _get_paddle_engine()
        results = engine.detect(process_path)
        
        if not results:
            return []
        
        # Search for text
        search_lower = search_text.lower().strip()
        matches = []
        
        for detection in results:
            coords, (text, confidence) = detection
            
            if confidence < min_confidence:
                continue
            
            if search_lower in text.lower():
                # Extract bounding box
                points = np.array(coords, dtype=np.int32)
                x_min = int(np.min(points[:, 0]))
                y_min = int(np.min(points[:, 1]))
                x_max = int(np.max(points[:, 0]))
                y_max = int(np.max(points[:, 1]))
                
                matches.append(WordBox(
                    text=text,
                    left=x_min,
                    top=y_min,
                    width=x_max - x_min,
                    height=y_max - y_min,
                    confidence=confidence
                ))
        
        return matches
        
    finally:
        # Cleanup temp file
        if preprocess and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


def get_all_text_paddle(image_path: str, preprocess: bool = True) -> List[WordBox]:
    """
    Get all detected text regions using PaddleOCR
    
    Args:
        image_path: Path to image file
        preprocess: Apply preprocessing for better accuracy
        
    Returns:
        List of all detected WordBox objects
    """
    # Preprocess if enabled
    if preprocess:
        preprocessed = preprocess_image(image_path, method="light")
        temp_path = image_path.replace('.', '_paddle_temp.')
        cv2.imwrite(temp_path, preprocessed)
        process_path = temp_path
    else:
        process_path = image_path
    
    try:
        # Run OCR
        engine = _get_paddle_engine()
        results = engine.detect(process_path)
        
        if not results:
            return []
        
        all_text = []
        for detection in results:
            coords, (text, confidence) = detection
            
            # Extract bounding box
            points = np.array(coords, dtype=np.int32)
            x_min = int(np.min(points[:, 0]))
            y_min = int(np.min(points[:, 1]))
            x_max = int(np.max(points[:, 0]))
            y_max = int(np.max(points[:, 1]))
            
            all_text.append(WordBox(
                text=text,
                left=x_min,
                top=y_min,
                width=x_max - x_min,
                height=y_max - y_min,
                confidence=confidence
            ))
        
        return all_text
        
    finally:
        # Cleanup temp file
        if preprocess and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


# ============================================================================
# TESSERACT ENGINE
# ============================================================================

def _setup_tesseract():
    """Setup Tesseract path automatically"""
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return False, None
    
    # Common paths
    paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'.format(
            os.getenv('USERNAME', '')
        ),
    ]
    
    # Check if already in PATH
    try:
        pytesseract.get_tesseract_version()
        return True, pytesseract
    except:
        pass
    
    # Try common paths
    for path in paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            try:
                pytesseract.get_tesseract_version()
                return True, pytesseract
            except:
                continue
    
    return False, None


_TESSERACT_AVAILABLE, _pytesseract = _setup_tesseract()


def find_text_with_tesseract(
    image_path: str,
    search_text: str,
    preprocess: bool = True,
    exact_match: bool = False
) -> List[WordBox]:
    """
    Find text using Tesseract OCR with optional preprocessing
    
    Args:
        image_path: Path to image file
        search_text: Text to search for
        preprocess: Apply preprocessing for better accuracy
        exact_match: Require exact word match
        
    Returns:
        List of WordBox objects with detected text regions
    """
    if not _TESSERACT_AVAILABLE:
        raise RuntimeError(
            "Tesseract not installed. Install from:\n"
            "https://github.com/UB-Mannheim/tesseract/wiki"
        )
    
    from PIL import Image
    from pytesseract import Output
    
    # Preprocess if enabled
    if preprocess:
        preprocessed = preprocess_image(image_path, method="aggressive")
        temp_path = image_path.replace('.', '_tess_temp.')
        cv2.imwrite(temp_path, preprocessed)
        process_path = temp_path
    else:
        process_path = image_path
    
    try:
        # Run OCR
        img = Image.open(process_path)
        config = '--oem 1 --psm 3'  # LSTM engine, auto page segmentation
        data = _pytesseract.image_to_data(img, output_type=Output.DICT, config=config)
        
        # Search for text
        search_tokens = search_text.lower().split()
        matches = []
        
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            if int(data['conf'][i]) < 0:
                continue
            
            text = data['text'][i].strip()
            if not text:
                continue
            
            # Check match
            text_lower = text.lower()
            if exact_match:
                if text_lower in search_tokens:
                    matches.append(WordBox(
                        text=text,
                        left=data['left'][i],
                        top=data['top'][i],
                        width=data['width'][i],
                        height=data['height'][i],
                        confidence=float(data['conf'][i]) / 100.0
                    ))
            else:
                for search_token in search_tokens:
                    if search_token in text_lower or text_lower in search_token:
                        matches.append(WordBox(
                            text=text,
                            left=data['left'][i],
                            top=data['top'][i],
                            width=data['width'][i],
                            height=data['height'][i],
                            confidence=float(data['conf'][i]) / 100.0
                        ))
                        break
        
        return matches
        
    finally:
        # Cleanup temp file
        if preprocess and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


# ============================================================================
# GPT VISION API
# ============================================================================

def evaluate_with_gpt(
    image_path: str,
    question_answer_pairs: Optional[List[Tuple[str, str]]] = None,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    preprocess: bool = True
) -> Dict[str, Any]:
    """
    Evaluate worksheet using OpenAI GPT Vision API
    
    Args:
        image_path: Path to worksheet image
        question_answer_pairs: List of (question, answer) tuples (Optional for auto-detection)
        api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        model: GPT model to use
        preprocess: Apply light preprocessing for better recognition
        
    Returns:
        Dictionary with evaluation results
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Install OpenAI: pip install openai")
    
    # Get API key
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY environment variable")
    
    # Preprocess if enabled
    if preprocess:
        preprocessed = preprocess_image(image_path, method="light")
        temp_path = image_path.replace('.', '_gpt_temp.')
        cv2.imwrite(temp_path, preprocessed)
        process_path = temp_path
    else:
        process_path = image_path
    
    try:
        # Encode image
        with open(process_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        # Create prompt
        if question_answer_pairs:
            qa_text = "\n".join([
                f"Q{i+1}: {q}\nExpected: {a}"
                for i, (q, a) in enumerate(question_answer_pairs)
            ])
            prompt = f"""Analyze this worksheet and evaluate answers.

Questions and expected answers:
{qa_text}

For each question provide:
1. Student's answer (text in image)
2. Whether correct (true/false)

Respond in JSON:
{{
    "q1": {{
        "question": "...",
        "isAnswerCorrect": true
    }},
    ...
}}"""
        else:
            prompt = """
You are a STRICT mathematics examiner evaluating a scanned worksheet image.

Your job is to detect questions, read student answers, compute the correct
mathematical result, and compare them.

------------------------------------------------
IMAGE ANALYSIS RULES
------------------------------------------------
1. Detect each question and its answer.
2. Questions are printed text beginning with Q1, Q2, etc.
3. The answer belongs to the nearest question ABOVE it.
4. Any text between one question and the next belongs to that question.
5. Separator lines or spacing do NOT break association.

------------------------------------------------
QUESTION TEXT RULES
------------------------------------------------
- Copy EXACT words from the START of each question.
- Maximum 5 words.
- Stop before punctuation (?, :, .).
- Do NOT rewrite or summarize.
- Each question label must be unique.

------------------------------------------------
ANSWER DETECTION RULES
------------------------------------------------
For every question:
- Extract exactly what the student wrote.
- Preserve math symbols if visible.
- If multiple lines exist, combine into one line.

------------------------------------------------
MATH EVALUATION RULES (CRITICAL)
------------------------------------------------
You MUST solve the math problem yourself.

DO NOT compare strings directly.

Normalize equivalents:
×, x, * → multiplication
½, 1/2 → same value
Ignore spacing differences.

Procedure:
1. Understand the math question.
2. Compute the correct answer.
3. Compare with detected student answer.

If mathematically equal → true
Otherwise → false

Example:
7 × 9 = 63 → correct
Area = bh/2 → correct form
2 × (8 + 5) = 26 → correct

Never mark a mathematically correct answer as false.

------------------------------------------------
OUTPUT FORMAT (STRICT JSON ONLY)
------------------------------------------------
Return ONLY valid JSON.

For each question return:

{
 "q1":{
   "question":"...",
   "detectedAnswer":"what you read from image",
   "expectedAnswer":"correct mathematical answer",
   "isAnswerCorrect":true
 }
}

No explanations.
No markdown.
No extra text.
"""

        
        # Call API
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=2000,
        )
        
        # Parse response
        content = response.choices[0].message.content

        print("\n[DEBUG] Raw GPT response content:")
        print(content)

        # Remove markdown code blocks
        if "```" in content:
            content = re.sub(r"```json\s*", "", content, flags=re.IGNORECASE)
            content = re.sub(r"```\s*", "", content, flags=re.IGNORECASE)
        
        content = content.strip()
        
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"\n[ERROR] Failed to parse GPT response as JSON:")
            print(f"--- RAW RESPONSE START ---\n{content}\n--- RAW RESPONSE END ---")
            print(f"Error details: {e}")
            return {}
        
    finally:
        # Cleanup temp file
        if preprocess and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def draw_boxes_on_image(
    image_path: str,
    boxes: List[WordBox],
    output_path: str,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    show_text: bool = True
) -> str:
    """
    Draw bounding boxes on image
    
    Args:
        image_path: Path to input image
        boxes: List of WordBox objects to draw
        output_path: Path to save annotated image
        color: Box color (B, G, R)
        thickness: Line thickness
        show_text: Show detected text above boxes
        
    Returns:
        Path to output image
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")
    
    for box in boxes:
        # Draw rectangle
        cv2.rectangle(
            img,
            (box.left, box.top),
            (box.right, box.bottom),
            color,
            thickness
        )
        
        # Draw text
        if show_text:
            label = f"{box.text} ({box.confidence:.2f})" if box.confidence > 0 else box.text
            cv2.putText(
                img,
                label,
                (box.left, box.top - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )
    
    cv2.imwrite(output_path, img)
    return output_path


def locate_and_annotate(
    image_path: str,
    gpt_response: Dict[str, Any],
    output_path: Optional[str] = None
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Locate questions using OCR and annotate with Correct/Wrong based on GPT result.
    
    Args:
        image_path: Path to the image
        gpt_response: JSON response from GPT
        output_path: Path to save annotated image
        
    Returns:
        Tuple[str, List[Dict]]: Path to saved image, List of annotation data [{'type': 'correct'|'wrong', 'x': int, 'y': int}]
    """
    if output_path is None:
        output_path = image_path.replace('.', '_graded.')
        
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")
    
    print("\nAnnotating image based on GPT results...")
    
    annotations = []
    
    # --- RUN OCR ONCE FOR THE ENTIRE PAGE ---
    # PaddleOCR already scans the whole image in one pass.
    # We run it once here and reuse the results for all question lookups,
    # instead of re-running OCR 9 times (once per question).
    print("  [OCR] Scanning entire page once...")
    preprocessed = preprocess_image(image_path, method="light")
    temp_path = image_path.replace('.', '_paddle_temp.')
    cv2.imwrite(temp_path, preprocessed)
    
    page_boxes: List[WordBox] = []
    ocr_scan_ok = False
    
    try:
        engine = _get_paddle_engine()
        all_ocr_results = engine.detect(temp_path) or []
        
        for detection in all_ocr_results:
            coords, (text, confidence) = detection
            if confidence < 0.6:
                continue
            points = np.array(coords, dtype=np.int32)
            x_min = int(np.min(points[:, 0]))
            y_min = int(np.min(points[:, 1]))
            x_max = int(np.max(points[:, 0]))
            y_max = int(np.max(points[:, 1]))
            page_boxes.append(WordBox(
                text=text,
                left=x_min, top=y_min,
                width=x_max - x_min, height=y_max - y_min,
                confidence=confidence
            ))
        ocr_scan_ok = True
        print(f"  [OCR] Found {len(page_boxes)} text boxes on page.")
    except Exception as ocr_err:
        print(f"  [OCR] Single-pass scan failed ({ocr_err}). Will fall back to per-question mode.")
    finally:
        try:
            os.remove(temp_path)
        except:
            pass
    # -----------------------------------------
    
    for key, item in gpt_response.items():
        question_text = item.get("question", "")
        is_correct = item.get("isAnswerCorrect", False)
        
        if not question_text:
            continue
            
        print(f"  Locating: '{question_text[:30]}...' -> {'Correct' if is_correct else 'Wrong'}")
        
        # Search cached OCR results (no re-scan)
        # If the single-pass scan failed, fall back to per-question OCR (old behavior)
        search_query = " ".join(question_text.split()[:5]).lower().strip()
        if ocr_scan_ok:
            boxes = [b for b in page_boxes if search_query in b.text.lower()]
        else:
            # FALLBACK: independent OCR per question (same as old behavior)
            boxes = find_text_with_paddle(image_path, search_query, preprocess=True, min_confidence=0.6)
        
        if boxes:
            # Use the first match
            box = boxes[0]
            
            # Choose color
            color = (0, 255, 0) if is_correct else (0, 0, 255) # Green / Red
            label = "Correct" if is_correct else "Wrong"
            # ERROR FIX: OpenCV FONT_HERSHEY_SIMPLEX doesn't support unicode icons like ✓/✗
            # Replacing with ASCII characters
            icon = "[/]" if is_correct else "[X]"
            
            # Draw box around question (optional, maybe just next to it)
            # cv2.rectangle(img, (box.left, box.top), (box.right, box.bottom), color, 2)
            
            # proper visual: Icon + Text next to the question
            # Calculate position: Right of the question
            text_x = box.right + 10
            text_y = box.top + 20
            
            # Ensure it fits in image
            if text_x > img.shape[1] - 100:
                text_x = box.left
                text_y = box.top - 10
            
            # Background for text
            (w, h), _ = cv2.getTextSize(f"{icon} {label}", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(img, (text_x - 5, text_y - h - 5), (text_x + w + 5, text_y + 5), (255, 255, 255), -1)
            
            # Draw text
            cv2.putText(
                img,
                f"{icon} {label}",
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

            # Collect annotation for handwriting system
            annotations.append({
                "type": "correct" if is_correct else "wrong",
                "x": int(text_x), # Store as standard Python int
                "y": int(text_y)
            })
        else:
            print(f"    Could not locate text: {search_query}")
            
    # --- ADD SCORE TO IMAGE ---
    if annotations:
        correct_count = sum(1 for a in annotations if a['type'] == 'correct')
        total_count = len(annotations)
        score_str = f"Score: {correct_count}/{total_count}"
        
        # Draw Score at top right
        # Background
        (sw, sh), _ = cv2.getTextSize(score_str, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4)
        cv2.rectangle(img, (img.shape[1] - sw - 40, 20), (img.shape[1] - 20, 20 + sh + 20), (255, 255, 255), -1)
        # Text
        cv2.putText(
            img,
            score_str,
            (img.shape[1] - sw - 30, 20 + sh + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            (255, 0, 0), # Blue
            4
        )
    # --------------------------

    cv2.imwrite(output_path, img)
    print(f"✓ Graded image saved: {output_path}")
    return output_path, annotations


def extract_text_with_layout(image_path: str, preprocess: bool = True) -> str:
    """
    Extract text from image attempting to preserve the original layout.
    
    Args:
        image_path: Path to the image
        preprocess: Whether to preprocess the image
        
    Returns:
        String containing the extracted text with layout preservation
    """
    boxes = get_all_text_paddle(image_path, preprocess=preprocess)
    
    if not boxes:
        return ""
        
    # Sort primarily by top (y), then by left (x)
    # This helps but isn't enough for true multi-column or complex layouts
    # A simple line-grouping algorithm is needed
    
    # Sort by Y first
    boxes.sort(key=lambda b: b.top)
    
    lines = []
    current_line = []
    current_y = boxes[0].top
    current_h = boxes[0].height
    
    # Threshold for considering text on the same line (e.g., half the line height)
    y_threshold = current_h * 0.5
    
    for box in boxes:
        # Check if this box belongs to the current line (vertical overlap)
        if abs(box.top - current_y) < y_threshold:
            current_line.append(box)
        else:
            # New line
            # Sort current line by X
            current_line.sort(key=lambda b: b.left)
            lines.append(current_line)
            
            # Start new line
            current_line = [box]
            current_y = box.top
            current_h = box.height
            y_threshold = current_h * 0.5
            
    # Append last line
    if current_line:
        current_line.sort(key=lambda b: b.left)
        lines.append(current_line)
        
    # Construct text
    output_text = []
    for line in lines:
        line_text = " ".join([box.text for box in line])
        output_text.append(line_text)
        
    return "\n".join(output_text)



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test OCR Engine")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("text", nargs="?", help="Text to search for (optional)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        exit(1)
        
    print(f"Loading image: {args.image}")
    
    if args.text:
        print(f"Searching for text: '{args.text}'")
        boxes = find_text_with_paddle(args.image, args.text, preprocess=True)
        
        if boxes:
            print(f"Found {len(boxes)} matches:")
            for i, box in enumerate(boxes):
                print(f"  {i+1}. '{box.text}' (Conf: {box.confidence:.2f}) at ({box.left}, {box.top})")
                
            # Draw boxes
            output_path = args.image.replace('.', '_found.')
            draw_boxes_on_image(args.image, boxes, output_path)
            print(f"Saved annotated image to: {output_path}")
        else:
            print(f"Text '{args.text}' not found.")
            
    else:
        print("Extracting all text...")
        boxes = get_all_text_paddle(args.image, preprocess=True)
        for box in boxes:
            print(f"  - '{box.text}' (Conf: {box.confidence:.2f})")
