"""
test_box_detection_live.py
--------------------------
Real-time live camera test for black-bordered box detection.
Point the camera at the question paper to see detected boxes highlighted in green.

Controls:
    Q / ESC  - Quit
    +/-      - Increase/decrease min box size
    [/]      - Increase/decrease max box size
    D        - Toggle debug print in terminal
"""

import cv2
from box_detector import detect_marking_boxes, draw_detected_boxes

# ── Config ────────────────────────────────────────────────────────────────────
CAMERA_ID   = 0     # Change to 1 or 2 if needed for your camera
MIN_SIZE    = 30    # Min box side length in pixels  (press +/- to tune live)
MAX_SIZE    = 300   # Max box side length in pixels  (press [/] to tune live)
# ──────────────────────────────────────────────────────────────────────────────


def main():
    global MIN_SIZE, MAX_SIZE

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {CAMERA_ID}")
        return

    print("=== Live Box Detection ===")
    print("  Q / ESC  → Quit")
    print("  +/-      → Min size ±5px")
    print("  [/]      → Max size ±10px")
    print("  D        → Toggle debug output")
    print()

    debug = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Cannot read from camera.")
            break

        # Detect boxes
        boxes = detect_marking_boxes(
            frame,
            min_size=MIN_SIZE,
            max_size=MAX_SIZE,
            approx_square=True,
            debug=debug
        )

        # Draw overlay
        display = draw_detected_boxes(frame, boxes)

        # Show current thresholds in corner
        cv2.putText(display,
                    f"min={MIN_SIZE}px  max={MAX_SIZE}px  boxes={len(boxes)}",
                    (10, display.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        cv2.imshow("Box Detection (Q to quit)", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):   # Q or ESC
            break
        elif key == ord('+') or key == ord('='):
            MIN_SIZE = min(MIN_SIZE + 5, MAX_SIZE - 10)
            print(f"min_size → {MIN_SIZE}")
        elif key == ord('-'):
            MIN_SIZE = max(5, MIN_SIZE - 5)
            print(f"min_size → {MIN_SIZE}")
        elif key == ord(']'):
            MAX_SIZE = min(MAX_SIZE + 10, 2000)
            print(f"max_size → {MAX_SIZE}")
        elif key == ord('['):
            MAX_SIZE = max(MIN_SIZE + 10, MAX_SIZE - 10)
            print(f"max_size → {MAX_SIZE}")
        elif key == ord('d'):
            debug = not debug
            print(f"debug → {debug}")

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
