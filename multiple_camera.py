import cv2
import numpy as np
import os
import time
from datetime import datetime

# Configuration
STABLE_FRAMES_THRESHOLD = 30  # Number of consecutive frames with valid contour before capture (increased from 5)
COOLDOWN_SECONDS = 3  # Wait time after capture before detecting again
MIN_CONTOUR_AREA_PERCENTAGE = 0.2  # Minimum area as percentage of frame (replaces MIN_CONTOUR_AREA)
MAX_CONTOUR_AREA_RATIO = 0.95  # Maximum ratio of contour area to frame area
DEBUG_MODE = False  # Set to True to see intermediate detection steps

def create_output_folder():
    """Create folder to save captured images"""
    folder_name = "captured_copies"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # Also create cropped subfolder
    cropped_folder = os.path.join(folder_name, "cropped")
    if not os.path.exists(cropped_folder):
        os.makedirs(cropped_folder)
    
    return folder_name

def order_points(pts):
    """Order points in top-left, top-right, bottom-right, bottom-left order"""
    rect = np.zeros((4, 2), dtype="float32")
    
    # Sum and difference to find corners
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    
    rect[0] = pts[np.argmin(s)]      # Top-left
    rect[2] = pts[np.argmax(s)]      # Bottom-right
    rect[1] = pts[np.argmin(diff)]   # Top-right
    rect[3] = pts[np.argmax(diff)]   # Bottom-left
    
    return rect

def four_point_transform(image, pts):
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

def is_rectangle_like(contour):
    """Check if a 4-point contour is rectangle-like by checking angles"""
    if len(contour) != 4:
        return False
    
    points = contour.reshape(4, 2)
    
    # Calculate angles between consecutive sides
    angles = []
    for i in range(4):
        p1 = points[i]
        p2 = points[(i + 1) % 4]
        p3 = points[(i + 2) % 4]
        
        # Vectors
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Angle between vectors
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        angle_deg = np.degrees(angle)
        angles.append(angle_deg)
    
    # Check if all angles are close to 90 degrees (allow 40-140 degree range)
    for angle in angles:
        if not (40 < angle < 140):
            return False
    
    return True

def crop_and_save(image, contour, camera_name, output_folder, camera_capture=None):
    """Crop the detected area and save images
    
    Args:
        image: The processing frame (for display)
        contour: Detected contour coordinates
        camera_name: Name of the camera
        output_folder: Output directory
        camera_capture: cv2.VideoCapture object to capture fresh high-res image
    """
    # If camera_capture is provided, capture a fresh high-resolution image
    if camera_capture is not None:
        ret, high_res_frame = camera_capture.read()
        if ret and high_res_frame is not None:
            # Scale the contour coordinates if processing and capture resolutions differ
            proc_h, proc_w = image.shape[:2]
            high_h, high_w = high_res_frame.shape[:2]
            
            scale_x = high_w / proc_w
            scale_y = high_h / proc_h
            
            # Scale contour to high-res image
            scaled_contour = contour.copy().astype(np.float32)
            scaled_contour[:, :, 0] *= scale_x
            scaled_contour[:, :, 1] *= scale_y
            scaled_contour = scaled_contour.astype(np.int32)
            
            # Use high-res image for saving
            save_image = high_res_frame
            save_contour = scaled_contour
        else:
            # Fallback to processed image if capture fails
            save_image = image
            save_contour = contour
    else:
        # No camera provided, use processed image
        save_image = image
        save_contour = contour
    
    # Create output with contour drawn on high-res image
    output = save_image.copy()
    cv2.drawContours(output, [save_contour], -1, (0, 255, 0), 3)
    
    # Apply perspective transform to get cropped copy from high-res image
    pts = save_contour.reshape(4, 2)
    cropped = four_point_transform(save_image, pts)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save images with maximum quality
    contour_path = os.path.join(output_folder, f"{camera_name}_contour_{timestamp}.png")
    cropped_path = os.path.join(output_folder, "cropped", f"{camera_name}_cropped_{timestamp}.png")
    
    # Use PNG for lossless compression
    cv2.imwrite(contour_path, output, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    cv2.imwrite(cropped_path, cropped, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    
    print(f"[{camera_name}] ✓ Captured and saved HIGH-RES images:")
    print(f"  - Resolution: {save_image.shape[1]}x{save_image.shape[0]}")
    print(f"  - {camera_name}_contour_{timestamp}.png")
    print(f"  - cropped/{camera_name}_cropped_{timestamp}.png")
    
    # Return display versions (from processed frame for preview)
    display_output = image.copy()
    cv2.drawContours(display_output, [contour], -1, (0, 255, 0), 3)
    display_cropped = four_point_transform(image, contour.reshape(4, 2))
    
    return display_output, display_cropped

class CameraHandler:
    """Handle individual camera with independent contour detection and capture"""
    def __init__(self, camera_id, camera_name, output_folder, camera_capture):
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.output_folder = output_folder
        self.camera_capture = camera_capture  # Store camera object for high-res capture
        self.stable_frames = 0
        self.last_capture_time = 0
        self.captured_image = None
        self.cropped_image = None
        self.debug_images = {}
        
    def process_frame(self, frame):
        """Process frame and detect stable contours"""
        current_time = time.time()
        
        # Resize frame for faster processing (optional - adjust PROCESS_WIDTH as needed)
        PROCESS_WIDTH = 640  # Process at lower resolution for speed
        h, w = frame.shape[:2]
        process_scale = PROCESS_WIDTH / w
        process_frame = cv2.resize(frame, None, fx=process_scale, fy=process_scale, interpolation=cv2.INTER_AREA)
        
        # Create display frame from processed size
        display_frame = process_frame.copy()
        
        # Check if in cooldown period
        if current_time - self.last_capture_time < COOLDOWN_SECONDS:
            cooldown_remaining = COOLDOWN_SECONDS - (current_time - self.last_capture_time)
            status_text = f"Cooldown: {cooldown_remaining:.1f}s"
            color = (0, 165, 255)  # Orange
            self.draw_status(display_frame, status_text, color, self.stable_frames)
            return display_frame
        
        # Detect contour on the processed (smaller) frame
        if DEBUG_MODE:
            contour, self.debug_images = detect_contour(process_frame, return_debug=True)
        else:
            contour = detect_contour(process_frame, return_debug=False)
        
        if contour is not None:
            # Draw contour on display frame
            cv2.drawContours(display_frame, [contour], -1, (0, 255, 0), 2)
            
            # Increment stable frames counter
            self.stable_frames += 1
            
            # Draw status
            status_text = f"Detecting... {self.stable_frames}/{STABLE_FRAMES_THRESHOLD}"
            color = (0, 255, 255)  # Yellow
            
            # Check if reached threshold
            if self.stable_frames >= STABLE_FRAMES_THRESHOLD:
                # Capture and save - will capture fresh high-res image from camera
                # Pass the processed frame and contour, function will handle scaling
                self.captured_image, self.cropped_image = crop_and_save(
                    process_frame, contour, self.camera_name, self.output_folder, self.camera_capture
                )
                
                # Reset and start cooldown
                self.stable_frames = 0
                self.last_capture_time = current_time
                
                status_text = "CAPTURED!"
                color = (0, 255, 0)  # Green
            
            self.draw_status(display_frame, status_text, color, self.stable_frames)
            return display_frame
        else:
            # No contour detected, reset counter
            self.stable_frames = 0
            status_text = "Searching for copy..."
            color = (0, 0, 255)  # Red
            self.draw_status(display_frame, status_text, color, self.stable_frames)
            return display_frame
    
    def draw_status(self, frame, text, color, progress):
        """Draw status bar and progress on frame"""
        h, w = frame.shape[:2]
        
        # Draw semi-transparent overlay at top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Draw status text
        cv2.putText(frame, f"[{self.camera_name}]", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, text, (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw progress bar
        if progress > 0:
            bar_width = int((w - 20) * (progress / STABLE_FRAMES_THRESHOLD))
            cv2.rectangle(frame, (10, h - 20), (10 + bar_width, h - 10), color, -1)
            cv2.rectangle(frame, (10, h - 20), (w - 10, h - 10), (255, 255, 255), 2)

def main():
    global DEBUG_MODE, MAX_CONTOUR_AREA_RATIO, MIN_CONTOUR_AREA_PERCENTAGE
    
    # Create output folder
    output_folder = create_output_folder()
    
    # Initialize cameras
    print("=" * 70)
    print("AUTOMATIC COPY DETECTION AND CAPTURE SYSTEM")
    print("=" * 70)
    print("\nInitializing cameras...")
    cap1 = cv2.VideoCapture(0)  # First camera
    cap2 = cv2.VideoCapture(1)  # Second camera
    
    # Check if cameras opened successfully
    if not cap1.isOpened():
        print("Error: Cannot open camera 1")
        return
    
    if not cap2.isOpened():
        print("Error: Cannot open camera 2")
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
    actual_width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"✓ Camera 1: {actual_width1}x{actual_height1}")
    print(f"✓ Camera 2: {actual_width2}x{actual_height2}")
    print("✓ Cameras initialized successfully!")
    print("\nConfiguration:")
    print(f"  - Stable frames threshold: {STABLE_FRAMES_THRESHOLD} frames")
    print(f"  - Cooldown period: {COOLDOWN_SECONDS} seconds")
    print(f"  - Minimum contour area: {int(MIN_CONTOUR_AREA_PERCENTAGE * 100)}% of frame")
    print(f"  - Maximum contour area: {int(MAX_CONTOUR_AREA_RATIO * 100)}% of frame")
    print(f"  - Output folder: {output_folder}/")
    print(f"  - Debug mode: {'ON' if DEBUG_MODE else 'OFF'}")
    print("\nControls:")
    print("  Press 'q' or ESC - Quit")
    print("  Press 'd' - Toggle Debug Mode (shows detection steps)")
    print("  Press '+' - Increase min contour area %")
    print("  Press '-' - Decrease min contour area %")
    print("  Press ']' - Increase max area ratio")
    print("  Press '[' - Decrease max area ratio")
    print("\nBoth cameras running independently. Place copy in view...")
    print("Using Otsu thresholding + Canny edge detection (proven method)")
    print("=" * 70)
    print()
    
    # Create camera handlers with camera objects for high-res capture
    cam1_handler = CameraHandler(0, "cam1", output_folder, cap1)
    cam2_handler = CameraHandler(1, "cam2", output_folder, cap2)
    
    try:
        while True:
            # Capture frames from both cameras
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            
            if not ret1 or not ret2:
                print("Error: Cannot read from cameras")
                break
            
            # Process frames independently
            processed_frame1 = cam1_handler.process_frame(frame1)
            processed_frame2 = cam2_handler.process_frame(frame2)
            
            # Add instruction overlay on one of the windows
            instruction_img = processed_frame1.copy()
            h, w = instruction_img.shape[:2]
            cv2.putText(instruction_img, "Click this window then press keys!", (10, h-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.imshow('Camera 1 - Live Detection', instruction_img)
            
            # Display both camera feeds with status
            cv2.imshow('Camera 2 - Live Detection', processed_frame2)
            
            # Show debug windows if debug mode is on
            if DEBUG_MODE:
                if hasattr(cam1_handler, 'debug_images') and cam1_handler.debug_images:
                    for name, img in cam1_handler.debug_images.items():
                        cv2.imshow(f'Cam1 Debug - {name}', img)
                
                if hasattr(cam2_handler, 'debug_images') and cam2_handler.debug_images:
                    for name, img in cam2_handler.debug_images.items():
                        cv2.imshow(f'Cam2 Debug - {name}', img)
            
            # Show last captured images if available
            if cam1_handler.cropped_image is not None:
                cv2.imshow('Camera 1 - Last Capture', cam1_handler.cropped_image)
            
            if cam2_handler.cropped_image is not None:
                cv2.imshow('Camera 2 - Last Capture', cam2_handler.cropped_image)
            
            # Wait for key press (1ms delay for real-time processing)
            key = cv2.waitKey(1) & 0xFF
            
            # Debug: Print key if any key is pressed (remove after testing)
            if key != 255:  # 255 means no key pressed
                print(f"[DEBUG] Key pressed: {key} ('{chr(key) if 32 <= key < 127 else '?'}')")
            
            # Toggle debug mode on 'd'
            if key == ord('d'):
                DEBUG_MODE = not DEBUG_MODE
                print(f"\n[INFO] Debug mode: {'ON' if DEBUG_MODE else 'OFF'}")
                if not DEBUG_MODE:
                    # Close debug windows
                    for i in range(1, 8):
                        try:
                            cv2.destroyWindow(f'Cam1 Debug - {i}_*')
                            cv2.destroyWindow(f'Cam2 Debug - {i}_*')
                        except:
                            pass
            
            # Adjust minimum contour area percentage
            elif key == ord('+') or key == ord('='):
                MIN_CONTOUR_AREA_PERCENTAGE = min(0.9, MIN_CONTOUR_AREA_PERCENTAGE + 0.05)
                print(f"\n[INFO] Minimum contour area increased to: {int(MIN_CONTOUR_AREA_PERCENTAGE * 100)}%")
            
            elif key == ord('-') or key == ord('_'):
                MIN_CONTOUR_AREA_PERCENTAGE = max(0.05, MIN_CONTOUR_AREA_PERCENTAGE - 0.05)
                print(f"\n[INFO] Minimum contour area decreased to: {int(MIN_CONTOUR_AREA_PERCENTAGE * 100)}%")
            
            # Adjust maximum contour area ratio
            elif key == ord(']'):
                MAX_CONTOUR_AREA_RATIO = min(0.95, MAX_CONTOUR_AREA_RATIO + 0.05)
                print(f"\n[INFO] Maximum contour area ratio increased to: {int(MAX_CONTOUR_AREA_RATIO * 100)}%")
            
            elif key == ord('['):
                MAX_CONTOUR_AREA_RATIO = max(0.10, MAX_CONTOUR_AREA_RATIO - 0.05)
                print(f"\n[INFO] Maximum contour area ratio decreased to: {int(MAX_CONTOUR_AREA_RATIO * 100)}%")
            
            # Quit on 'q' or ESC
            elif key == ord('q') or key == 27:
                print("\nExiting...")
                break
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user...")
    
    finally:
        # Release resources
        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()
        print("\n" + "=" * 70)
        print("Cameras released. All captures saved to '{}' folder".format(output_folder))
        print("=" * 70)
        print("Goodbye!")

if __name__ == "__main__":
    main()
