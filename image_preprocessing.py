"""
Image Preprocessing Module for Enhanced OCR Accuracy

This module provides image preprocessing functions to improve OCR accuracy
for both Tesseract and PaddleOCR engines. Preprocessing includes:
- Grayscale conversion
- Noise reduction
- Contrast enhancement
- Adaptive thresholding
- Deskewing

Usage:
    from image_preprocessing import preprocess_for_ocr
    
    # Automatic preprocessing (recommended)
    processed = preprocess_for_ocr('image.png')
    
    # Or save preprocessed image
    processed = preprocess_for_ocr('image.png', save_path='processed.png')
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, Optional, Tuple
import os


def preprocess_for_ocr(
    image_path: Union[str, Path, np.ndarray],
    save_path: Optional[Union[str, Path]] = None,
    denoise: bool = True,
    enhance_contrast: bool = True,
    adaptive_threshold: bool = True,
    deskew: bool = True,
    resize_to: Optional[Tuple[int, int]] = None,
    debug: bool = False
    ) -> np.ndarray:
    """
    Preprocess image for optimal OCR accuracy
    
    Args:
        image_path: Path to image file or numpy array
        save_path: Optional path to save preprocessed image
        denoise: Apply denoising filter
        enhance_contrast: Apply CLAHE contrast enhancement
        adaptive_threshold: Apply adaptive thresholding (best for printed text)
        deskew: Automatically correct text rotation
        resize_to: Optional (width, height) to resize to
        debug: Show intermediate steps
        
    Returns:
        Preprocessed image as numpy array (grayscale)
    """
    # Load image
    if isinstance(image_path, np.ndarray):
        img = image_path
    else:
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")
    
    original = img.copy()
    
    # Step 1: Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    if debug:
        cv2.imshow('1. Grayscale', gray)
    
    # Step 2: Resize if requested (before other processing)
    if resize_to:
        gray = cv2.resize(gray, resize_to, interpolation=cv2.INTER_CUBIC)
        if debug:
            cv2.imshow('2. Resized', gray)
    
    # Step 3: Denoise
    if denoise:
        # Use Non-Local Means Denoising - excellent for OCR
        gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
        if debug:
            cv2.imshow('3. Denoised', gray)
    
    # Step 4: Enhance contrast with CLAHE
    if enhance_contrast:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        if debug:
            cv2.imshow('4. Contrast Enhanced', gray)
    
    # Step 5: Deskew if needed
    if deskew:
        gray = deskew_image(gray)
        if debug:
            cv2.imshow('5. Deskewed', gray)
    
    # Step 6: Adaptive thresholding for clearest text
    if adaptive_threshold:
        # Try both methods and use the one with better text clarity
        thresh1 = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Alternative: Otsu's binarization (works well for uniform lighting)
        _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Use the version with more white (assuming text is darker)
        if np.sum(thresh1 == 255) > np.sum(thresh2 == 255):
            processed = thresh1
        else:
            processed = thresh2
            
        if debug:
            cv2.imshow('6a. Adaptive Threshold', thresh1)
            cv2.imshow('6b. Otsu Threshold', thresh2)
            cv2.imshow('6c. Selected', processed)
    else:
        processed = gray
    
    # Save if path provided
    if save_path:
        cv2.imwrite(str(save_path), processed)
        print(f"✓ Preprocessed image saved to: {save_path}")
    
    if debug:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return processed


def deskew_image(image: np.ndarray) -> np.ndarray:
    """
    Automatically detect and correct text skew/rotation
    
    Args:
        image: Grayscale image
        
    Returns:
        Deskewed image
    """
    # Get binary image for line detection
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Detect lines using Hough transform
    coords = np.column_stack(np.where(binary > 0))
    
    if len(coords) < 10:
        return image  # Not enough points to determine angle
    
    # Calculate skew angle using minimum area rectangle
    angle = cv2.minAreaRect(coords)[-1]
    
    # Adjust angle
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90
    
    # Only correct if angle is significant (> 0.5 degrees)
    if abs(angle) < 0.5:
        return image
    
    # Rotate image to correct skew
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return rotated


def preprocess_light(image_path: Union[str, Path, np.ndarray]) -> np.ndarray:
    """
    Light preprocessing - just grayscale + denoise
    Good for high-quality scans or when you want to preserve original text
    
    Args:
        image_path: Path to image or numpy array
        
    Returns:
        Lightly preprocessed grayscale image
    """
    return preprocess_for_ocr(
        image_path,
        denoise=True,
        enhance_contrast=False,
        adaptive_threshold=False,
        deskew=False
    )


def preprocess_aggressive(image_path: Union[str, Path, np.ndarray]) -> np.ndarray:
    """
    Aggressive preprocessing - all filters enabled
    Best for poor quality images, photos, or handwritten text
    
    Args:
        image_path: Path to image or numpy array
        
    Returns:
        Heavily preprocessed image optimized for OCR
    """
    return preprocess_for_ocr(
        image_path,
        denoise=True,
        enhance_contrast=True,
        adaptive_threshold=True,
        deskew=True
    )


def batch_preprocess(
    input_folder: Union[str, Path],
    output_folder: Union[str, Path],
    pattern: str = "*.png",
    **kwargs
    ) -> int:
    """
    Preprocess all images in a folder
    
    Args:
        input_folder: Folder containing images
        output_folder: Folder to save preprocessed images
        pattern: File pattern (e.g., "*.png", "*.jpg")
        **kwargs: Arguments to pass to preprocess_for_ocr
        
    Returns:
        Number of images processed
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True, parents=True)
    
    count = 0
    for img_file in input_path.glob(pattern):
        try:
            processed = preprocess_for_ocr(img_file, **kwargs)
            save_path = output_path / img_file.name
            cv2.imwrite(str(save_path), processed)
            count += 1
            print(f"✓ Processed: {img_file.name}")
        except Exception as e:
            print(f"✗ Error processing {img_file.name}: {e}")
    
    print(f"\n✓ Processed {count} images to {output_folder}")
    return count


def compare_preprocessing(image_path: Union[str, Path]) -> None:
    """
    Show side-by-side comparison of different preprocessing methods
    Useful for determining which preprocessing works best for your images
    
    Args:
        image_path: Path to test image
    """
    # Load original
    original = cv2.imread(str(image_path))
    if original is None:
        print(f"Error: Cannot load {image_path}")
        return
    
    # Get different versions
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    light = preprocess_light(original)
    aggressive = preprocess_aggressive(original)
    
    # Create comparison grid
    h, w = original.shape[:2]
    
    # Resize for display if too large
    max_width = 800
    if w > max_width:
        scale = max_width / w
        original = cv2.resize(original, None, fx=scale, fy=scale)
        gray = cv2.resize(gray, None, fx=scale, fy=scale)
        light = cv2.resize(light, None, fx=scale, fy=scale)
        aggressive = cv2.resize(aggressive, None, fx=scale, fy=scale)
    
    # Convert grayscale to BGR for stacking
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    light_bgr = cv2.cvtColor(light, cv2.COLOR_GRAY2BGR)
    aggressive_bgr = cv2.cvtColor(aggressive, cv2.COLOR_GRAY2BGR)
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(original, "Original (Color)", (10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(gray_bgr, "Grayscale Only", (10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(light_bgr, "Light (Grayscale + Denoise)", (10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(aggressive_bgr, "Aggressive (All Filters)", (10, 30), font, 1, (0, 255, 0), 2)
    
    # Stack images
    top_row = np.hstack([original, gray_bgr])
    bottom_row = np.hstack([light_bgr, aggressive_bgr])
    comparison = np.vstack([top_row, bottom_row])
    
    # Show
    cv2.imshow('Preprocessing Comparison - Press any key to close', comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\nPreprocessing Comparison:")
    print("  Top-left: Original color image")
    print("  Top-right: Grayscale only")
    print("  Bottom-left: Light preprocessing (recommended for good quality images)")
    print("  Bottom-right: Aggressive preprocessing (recommended for poor quality/photos)")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Image Preprocessing for OCR")
        print("\nUsage:")
        print("  python image_preprocessing.py <image_path> [output_path]")
        print("\nExamples:")
        print("  python image_preprocessing.py image.png")
        print("  python image_preprocessing.py image.png preprocessed.png")
        print("  python image_preprocessing.py image.png --compare")
        print("\nOptions:")
        print("  --compare    Show comparison of preprocessing methods")
        print("  --debug      Show intermediate processing steps")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        sys.exit(1)
    
    # Comparison mode
    if len(sys.argv) > 2 and sys.argv[2] == '--compare':
        compare_preprocessing(image_path)
    
    # Debug mode
    elif len(sys.argv) > 2 and sys.argv[2] == '--debug':
        print("Processing with debug mode (showing all steps)...")
        processed = preprocess_for_ocr(image_path, debug=True)
    
    # Normal processing with output
    else:
        output_path = sys.argv[2] if len(sys.argv) > 2 else 'preprocessed.png'
        
        print(f"Preprocessing: {image_path}")
        print("Applying: Grayscale → Denoise → Contrast → Deskew → Threshold")
        
        processed = preprocess_for_ocr(
            image_path,
            save_path=output_path,
            denoise=True,
            enhance_contrast=True,
            adaptive_threshold=True,
            deskew=True
        )
        
        print(f"\n✓ Done! Preprocessed image shape: {processed.shape}")
        print(f"✓ Saved to: {output_path}")
        print("\nTip: Use '--compare' to see different preprocessing levels")
