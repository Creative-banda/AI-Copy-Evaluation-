"""
test_box_detection_live.py
--------------------------
Test black-bordered box detection on a single image.

Usage:
    python test_box_detection_live.py <image_path>
    python test_box_detection_live.py captured_copies/cam1_xxx/original_gray.png

Controls (while window is open):
    Q / ESC  - Quit
    S        - Save annotated image
"""

import sys
import cv2
from box_detector import detect_marking_boxes, draw_detected_boxes

# ── Config ────────────────────────────────────────────────────────────────────
MIN_SIZE = 60    # Min box side length in pixels — increase if detecting noise/shadows
MAX_SIZE = 300   # Max box side length in pixels
# ──────────────────────────────────────────────────────────────────────────────


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_box_detection_live.py <image_path>")
        print("Example: python test_box_detection_live.py captured_copies/cam1_xxx/original_gray.png")
        sys.exit(1)

    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Cannot load image: {image_path}")
        sys.exit(1)

    print(f"Loaded: {image_path}  ({image.shape[1]}x{image.shape[0]}px)")

    # Detect boxes
    boxes = detect_marking_boxes(
        image,
        min_size=MIN_SIZE,
        max_size=MAX_SIZE,
        approx_square=True,
        debug=True
    )

    print(f"\nDetected {len(boxes)} marking box(es):")
    for i, b in enumerate(boxes):
        print(f"  #{i+1}  x={b['x']} y={b['y']} w={b['w']} h={b['h']}  center=({b['cx']}, {b['cy']})")

    # Draw overlay
    annotated = draw_detected_boxes(image, boxes)

    # Save annotated image alongside original
    save_path = image_path.rsplit('.', 1)[0] + "_boxes_detected.png"
    cv2.imwrite(save_path, annotated)
    print(f"\nSaved annotated image: {save_path}")

    # Resize for display if image is very large
    display = annotated
    h, w = annotated.shape[:2]
    if w > 1280:
        scale = 1280 / w
        display = cv2.resize(annotated, (1280, int(h * scale)))

    cv2.imshow("Box Detection (Q to quit, S to save)", display)
    print("Press Q/ESC to close window.")

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (ord('q'), 27):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

