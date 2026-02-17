"""
Camera System Module - Dual Camera Document Capture with Preprocessing

This module handles:
    - Dual camera initialization
    - Automatic document detection (Otsu + Canny)
    - Image preprocessing (grayscale conversion)
    - Cropping and saving (4 versions: color, grayscale, color-cropped, grayscale-cropped)

Export functions:
    - start_camera_system() - Main function to start dual camera capture
"""

import cv2
import numpy as np
import os
import time
from datetime import datetime
from typing import Tuple, Optional, Callable, Dict, Any


# Configuration
STABLE_FRAMES_THRESHOLD = 30
COOLDOWN_SECONDS = 3
MIN_CONTOUR_AREA_PERCENTAGE = 0.2
MAX_CONTOUR_AREA_RATIO = 0.95
PROCESS_WIDTH = 640  # Process at lower resolution for speed
DEBUG_MODE = False


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order points in top-left, top-right, bottom-right, bottom-left order"""
    rect = np.zeros((4, 2), dtype="float32")
    
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    
    rect[0] = pts[np.argmin(s)]      # Top-left
    rect[2] = pts[np.argmax(s)]      # Bottom-right
    rect[1] = pts[np.argmin(diff)]   # Top-right
    rect[3] = pts[np.argmax(diff)]   # Bottom-left
    
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply perspective transform to get top-down view"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Compute width
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # Compute height
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Destination points
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    # Perspective transform
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped


def is_reasonable_quad(points):
    """Check if the quadrilateral has reasonable proportions for a document"""
    if len(points) != 4:
        return False
        
    # Convert points to a more usable format
    pts = np.array([p[0] for p in points], dtype=np.float32)
    
    # Get width and height of bounding rect
    x, y, w, h = cv2.boundingRect(points)
    
    # Check aspect ratio - most documents have reasonable aspect ratios
    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
    if not (1.0 <= aspect_ratio <= 2.5):  # Common document aspect ratios
        return False
        
    # Check that points form a convex shape
    if not cv2.isContourConvex(points):
        return False
        
    # Calculate and check minimum angle between edges
    # Documents should have roughly 90 degree angles
    min_angle = 180
    for i in range(4):
        pt1 = pts[i]
        pt2 = pts[(i+1) % 4]
        pt3 = pts[(i+2) % 4]
        
        # Vectors for angle calculation
        v1 = pt1 - pt2
        v2 = pt3 - pt2
        
        # Calculate angle in degrees
        dot = np.dot(v1, v2)
        cross = np.cross(v1, v2)
        angle = np.degrees(np.arctan2(cross, dot))
        angle = abs(angle)
        
        min_angle = min(min_angle, angle)
    
    # Minimum angle should be close to 90 degrees for documents
    # Allow some flexibility (45-135 degrees)
    return min_angle >= 45


def detect_contour(image, return_debug=False):
    """Detect the copy/document edges using Otsu thresholding + Canny edges (from working demo)"""
    screenCnt = None
    debug_images = {}
    
    # Get frame dimensions
    frame_height, frame_width = image.shape[:2]
    frame_area = frame_height * frame_width
    min_allowed_area = MIN_CONTOUR_AREA_PERCENTAGE * frame_area
    max_allowed_area = frame_area * MAX_CONTOUR_AREA_RATIO
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    if return_debug:
        debug_images['1_gray_blur'] = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)
    
    # Apply Otsu's thresholding (original method from demo)
    _, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    if return_debug:
        debug_images['2_otsu_threshold'] = cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR)
    
    # Also try Canny edge detection - good for documents with less contrast
    edges = cv2.Canny(blur, 30, 200)
    
    if return_debug:
        debug_images['3_canny_edges'] = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Combine thresholded image and edges to improve detection
    combined = cv2.bitwise_or(threshold, edges)
    
    if return_debug:
        debug_images['4_combined'] = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
    
    # Dilate to connect broken lines in the edges
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(combined, kernel, iterations=1)
    
    if return_debug:
        debug_images['5_dilated'] = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
    
    # Find contours in both the threshold and dilated images
    contours1, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Combine contours from both methods
    contours = list(contours1) + list(contours2)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    if return_debug:
        # Draw all detected contours for debugging
        all_contours_img = image.copy()
        valid_contours = [c for c in contours if min_allowed_area < cv2.contourArea(c) < max_allowed_area]
        cv2.drawContours(all_contours_img, valid_contours[:10], -1, (0, 255, 255), 2)
        cv2.putText(all_contours_img, f"Valid contours: {len(valid_contours)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        debug_images['6_all_contours'] = all_contours_img
    
    if not contours:
        if return_debug:
            return None, debug_images
        return None
    
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Check if contour area is significant
        if area > min_allowed_area and area < max_allowed_area:
            peri = cv2.arcLength(contour, True)
            # Use a smaller epsilon for better approximation (from demo)
            epsilon = 0.02 * peri
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if we've found a quadrilateral (4 points)
            if len(approx) == 4:
                # Additional checks for document-like shapes
                if is_reasonable_quad(approx):
                    if area > max_area:
                        screenCnt = approx
                        max_area = area
                        
                        if return_debug:
                            x, y, w, h = cv2.boundingRect(approx)
                            found_img = image.copy()
                            cv2.drawContours(found_img, [screenCnt], -1, (0, 255, 0), 3)
                            # Draw corner points
                            for point in screenCnt:
                                px, py = point[0]
                                cv2.circle(found_img, (px, py), 5, (255, 0, 0), -1)
                            cv2.putText(found_img, f"Area: {int(area)}", (x, y-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            debug_images['7_found_document'] = found_img
    
    if return_debug:
        return screenCnt, debug_images
    return screenCnt


def apply_ocr_enhancement(image: np.ndarray, is_grayscale: bool = False) -> np.ndarray:
    """
    Apply best filters for OCR/AI processing
    
    Args:
        image: Input image (BGR or Grayscale)
        is_grayscale: Whether the image is already grayscale
        
    Returns:
        Enhanced image optimized for OCR
    """
    # Convert to grayscale if needed
    if not is_grayscale and len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy() if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. Denoise - Remove camera noise while preserving edges
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # 2. CLAHE - Adaptive contrast enhancement (best for uneven lighting)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # 3. Sharpen - Make text edges clearer
    kernel_sharpen = np.array([[-1,-1,-1],
                               [-1, 9,-1],
                               [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)
    
    # 4. Bilateral filter - Smooth while preserving edges (best for text)
    bilateral = cv2.bilateralFilter(sharpened, 9, 75, 75)
    
    return bilateral


def apply_color_enhancement(image: np.ndarray) -> np.ndarray:
    """
    Enhance color image for better OCR/AI processing
    
    Args:
        image: Input BGR image
        
    Returns:
        Enhanced color image
    """
    # 1. Denoise color image
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    
    # 2. Convert to LAB color space for better enhancement
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 3. Apply CLAHE to L channel (brightness)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    # 4. Merge back
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    # 5. Sharpen
    kernel_sharpen = np.array([[-1,-1,-1],
                               [-1, 9,-1],
                               [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)
    
    # 6. Bilateral filter
    bilateral = cv2.bilateralFilter(sharpened, 9, 75, 75)
    
    return bilateral


def crop_and_save(image, contour, camera_name, output_folder, camera_capture=None):
    """Crop the detected area and save images
    
    Args:
        image: The processing frame (ONLY for scaling calculations, NOT for saving)
        contour: Detected contour coordinates (from low-res processing frame)
        camera_name: Name of the camera
        output_folder: Output directory
        camera_capture: cv2.VideoCapture object to capture fresh high-res image
    """
    # MUST capture a fresh high-resolution image - NEVER use the processed frame for saving
    if camera_capture is None:
        print(f"ERROR: No camera provided, cannot save high-quality images")
        return None, None
    
    # Capture FRESH high-resolution frame directly from camera
    ret, high_res_frame = camera_capture.read()
    
    if not ret or high_res_frame is None:
        print(f"ERROR: Failed to capture high-res frame from camera")
        return None, None
    
    # Scale the contour coordinates from low-res processing frame to high-res capture frame
    proc_h, proc_w = image.shape[:2]
    high_h, high_w = high_res_frame.shape[:2]
    
    scale_x = high_w / proc_w
    scale_y = high_h / proc_h
    
    # Scale contour to match high-res image dimensions
    scaled_contour = contour.copy().astype(np.float32)
    scaled_contour[:, :, 0] *= scale_x
    scaled_contour[:, :, 1] *= scale_y
    scaled_contour = scaled_contour.astype(np.int32)
    
    # NOW work ONLY with the high-res frame - completely ignore the processed frame
    save_image = high_res_frame
    save_contour = scaled_contour
    
    print(f"\n[{camera_name}] CAPTURE INFO:")
    print(f"  Processing frame size: {proc_w}x{proc_h} (low-res for detection only)")
    print(f"  Captured frame size: {high_w}x{high_h} (HIGH-RES for saving)")
    print(f"  Scale factor: {scale_x:.2f}x")
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate timestamp for unique filenames and folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    capture_folder = os.path.join(output_folder, f"{camera_name}_{timestamp}")
    
    # Create capture-specific folder
    os.makedirs(capture_folder, exist_ok=True)
    
    # ========================================================================
    # SAVE ORIGINAL VERSIONS (No enhancement - preserve original quality)
    # ========================================================================
    
    # NOTE: Skipping full resolution frame saving as per requirements to save space
    # We only care about the cropped document
    
    # 3. Cropped document (color) - ORIGINAL
    pts = save_contour.reshape(4, 2)
    cropped_color = four_point_transform(save_image, pts)
    
    
    # Rotate based on camera
    # cam1: -90 degrees (Counter Clockwise)
    # cam2: +90 degrees (Clockwise)
    if "cam2" in str(camera_name).lower():
         cropped_color = cv2.rotate(cropped_color, cv2.ROTATE_90_CLOCKWISE)
    else:
         cropped_color = cv2.rotate(cropped_color, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    cropped_path_color = os.path.join(capture_folder, f"original_color.png")
    cv2.imwrite(cropped_path_color, cropped_color, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    
    # 4. Cropped (grayscale) - ORIGINAL
    cropped_gray = cv2.cvtColor(cropped_color, cv2.COLOR_BGR2GRAY)
    cropped_path_gray = os.path.join(capture_folder, f"original_gray.png")
    cv2.imwrite(cropped_path_gray, cropped_gray, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    
    # ========================================================================
    # SAVE GPT COMPRESSED VERSION (Optimized for Tokens)
    # ========================================================================
    # Resize to simpler resolution (e.g., max 2048px width) and compress as JPG
    
    h, w = cropped_color.shape[:2]
    max_dim = 2048
    scale = 1.0
    if w > max_dim or h > max_dim:
        scale = max_dim / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        gpt_image = cv2.resize(cropped_color, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        gpt_image = cropped_color.copy()
        
    gpt_compressed_path = os.path.join(capture_folder, "gpt_compressed.jpg")
    # Save as JPEG with 95% quality (High quality for text)
    cv2.imwrite(gpt_compressed_path, gpt_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    # Get actual file sizes
    size_cropped = os.path.getsize(cropped_path_color) / (1024 * 1024)  # MB
    size_gpt = os.path.getsize(gpt_compressed_path) / 1024  # KB
    
    print(f"[{camera_name}] ✓ Captured -> '{capture_folder}'")
    print(f"  • Cropped Original: {size_cropped:.2f} MB")
    print(f"  • GPT Compressed: {size_gpt:.1f} KB (Resized & Compressed)")
    
    # Return all paths and display images
    paths = {
        "cropped_color": cropped_path_color,
        "cropped_gray": cropped_path_gray,
        "gpt_compressed": gpt_compressed_path
    }

    # Return display versions (from processed frame for preview)
    # Use proper variable names from the start of the function
    display_output = image.copy()
    cv2.drawContours(display_output, [contour], -1, (0, 255, 0), 3)
    display_cropped = four_point_transform(image, contour.reshape(4, 2))
    
    return display_output, display_cropped, paths


class CameraHandler:
    """Handle individual camera with independent document detection"""
    
    def __init__(self, camera_id: int, camera_name: str, output_folder: str, camera_capture, 
                 on_capture: Optional[Callable[[Dict[str, str]], None]] = None):
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.output_folder = output_folder
        self.camera_capture = camera_capture
        self.on_capture = on_capture
        self.stable_frames = 0
        self.last_capture_time = 0
        self.saved_paths = None
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame and detect stable contours"""
        current_time = time.time()
        
        # Resize for faster processing
        h, w = frame.shape[:2]
        process_scale = PROCESS_WIDTH / w
        process_frame = cv2.resize(frame, None, fx=process_scale, fy=process_scale, 
                                   interpolation=cv2.INTER_AREA)
        
        display_frame = process_frame.copy()
        
        # Check cooldown
        if current_time - self.last_capture_time < COOLDOWN_SECONDS:
            cooldown_remaining = COOLDOWN_SECONDS - (current_time - self.last_capture_time)
            self._draw_status(display_frame, f"Cooldown: {cooldown_remaining:.1f}s", 
                            (0, 165, 255), self.stable_frames)
            return display_frame
        
        # Detect document
        if DEBUG_MODE:
            contour, self.debug_images = detect_contour(process_frame, return_debug=True)
        else:
            contour = detect_contour(process_frame, return_debug=False)
        
        if contour is not None:
            # Draw contour
            cv2.drawContours(display_frame, [contour], -1, (0, 255, 0), 2)
            
            # Increment stability counter
            self.stable_frames += 1
            
            status_text = f"Detecting... {self.stable_frames}/{STABLE_FRAMES_THRESHOLD}"
            color = (0, 255, 255)  # Yellow
            
            # Check if stable enough to capture
            if self.stable_frames >= STABLE_FRAMES_THRESHOLD:
                # Capture and save - will capture fresh high-res image from camera
                # Pass the processed frame and contour, function will handle scaling
                self.captured_image, self.cropped_image, saved_paths = crop_and_save(
                    process_frame, contour, self.camera_name, self.output_folder, self.camera_capture
                )
                
                # Reset and start cooldown
                self.stable_frames = 0
                self.last_capture_time = current_time
                
                status_text = "CAPTURED!"
                color = (0, 255, 0)  # Green
                
                # Trigger callback if defined
                if self.on_capture and saved_paths:
                    self.on_capture({
                        "camera_name": self.camera_name,
                        "paths": saved_paths
                    })
            
            self._draw_status(display_frame, status_text, color, self.stable_frames)
        else:
            # No contour detected
            self.stable_frames = 0
            self._draw_status(display_frame, "Searching for document...", 
                            (0, 0, 255), self.stable_frames)
        
        return display_frame
    
    def _draw_status(self, frame: np.ndarray, text: str, color: Tuple[int, int, int], 
                     progress: int):
        """Draw status overlay on frame"""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Status text
        cv2.putText(frame, f"[{self.camera_name}]", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, text, (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Progress bar
        if progress > 0:
            bar_width = int((w - 20) * (progress / STABLE_FRAMES_THRESHOLD))
            cv2.rectangle(frame, (10, h - 20), (10 + bar_width, h - 10), color, -1)
            cv2.rectangle(frame, (10, h - 20), (w - 10, h - 10), (255, 255, 255), 2)


def start_camera_system(
    camera1_id: int = 0,
    camera2_id: int = 1,
    output_folder: str = "captured_copies",
    on_capture: Optional[Callable[[Dict[str, str]], None]] = None
) -> None:
    """
    Start dual camera document capture system
    
    Args:
        camera1_id: First camera device ID
        camera2_id: Second camera device ID
        output_folder: Base folder to save captured documents
        
    Controls:
        Press 'q' or ESC - Quit
        Press 'd' - Toggle debug mode
    """
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize cameras
    print("=" * 70)
    print("DUAL CAMERA DOCUMENT CAPTURE SYSTEM")
    print("=" * 70)
    print("\nInitializing cameras...")
    
    cap1 = cv2.VideoCapture(camera1_id)
    cap2 = cv2.VideoCapture(camera2_id)
    
    if not cap1.isOpened():
        print(f"Error: Cannot open camera {camera1_id}")
        return
    
    if not cap2.isOpened():
        print(f"Error: Cannot open camera {camera2_id}")
        cap1.release()
        return
    
    # Set cameras to maximum resolution for best quality
    # Try common high resolutions (camera will use highest supported)
    for cap in [cap1, cap2]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # 1080p width
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # 1080p height
        # Try even higher if camera supports it
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)  # 4K width
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)  # 4K height
    
    # Get actual resolution
    w1, h1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w2, h2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"✓ Camera 1: {w1}x{h1}")
    print(f"✓ Camera 2: {w2}x{h2}")
    print("✓ Cameras initialized!")
    
    print("\nConfiguration:")
    print(f"  • Camera resolution: {w1}x{h1} (Camera 1), {w2}x{h2} (Camera 2)")
    print(f"  • Image format: PNG (Uncompressed, Lossless)")
    print(f"  • Stability threshold: {STABLE_FRAMES_THRESHOLD} frames")
    print(f"  • Cooldown period: {COOLDOWN_SECONDS} seconds")
    print(f"  • Output folder: {output_folder}/")
    print(f"  • Save formats: Color + Grayscale, Full + Cropped (4 versions)")
    
    print("\nControls:")
    print("  Press 'q' or ESC - Quit")
    print("  Press 'd' - Toggle Debug Mode")
    
    print("\n" + "=" * 70)
    print("Place documents in view of cameras...")
    print("=" * 70 + "\n")
    
    # Create camera handlers
    cam1_handler = CameraHandler(camera1_id, "cam1", output_folder, cap1, on_capture)
    cam2_handler = CameraHandler(camera2_id, "cam2", output_folder, cap2, on_capture)
    
    try:
        while True:
            # Capture frames
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            
            if not ret1 or not ret2:
                print("Error: Cannot read from cameras")
                break
            
            # Process frames
            processed1 = cam1_handler.process_frame(frame1)
            processed2 = cam2_handler.process_frame(frame2)
            
            # Display
            cv2.imshow('Camera 1 - Document Detection', processed1)
            cv2.imshow('Camera 2 - Document Detection', processed2)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Quit
                print("\nExiting...")
                break
            elif key == ord('d'):  # Toggle debug
                global DEBUG_MODE
                DEBUG_MODE = not DEBUG_MODE
                print(f"\nDebug mode: {'ON' if DEBUG_MODE else 'OFF'}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user...")
    
    finally:
        # Cleanup
        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 70)
        print(f"Cameras released. All captures saved to '{output_folder}' folder")
        print("=" * 70)


if __name__ == "__main__":
    print("Camera System Module")
    print("=" * 70)
    print("Usage: from camera_system import start_camera_system")
    print("       start_camera_system()")
    print("=" * 70)
    
    # Start if run directly
    start_camera_system()
