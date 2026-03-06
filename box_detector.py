"""
box_detector.py
---------------
Modular black-bordered rectangle detection for question paper marking boxes.

Exported functions:
    detect_marking_boxes(image, min_size, max_size, approx_square) -> list[dict]
    draw_detected_boxes(image, boxes) -> image (with overlays drawn)
"""

import cv2
import numpy as np


def detect_marking_boxes(
    image,
    min_size: int = 30,
    max_size: int = 300,
    approx_square: bool = True,
    aspect_tolerance: float = 0.6,
    debug: bool = False
) -> list:
    """
    Detect black-bordered rectangular marking boxes on a question paper.

    Args:
        image        : OpenCV BGR image (numpy array)
        min_size     : Minimum width/height of a box in pixels (tune for camera distance)
        max_size     : Maximum width/height of a box in pixels
        approx_square: If True, filter by near-square aspect ratio
        aspect_tolerance: Allowed deviation from 1:1 ratio (0.6 = width can be 0.6x to 1.4x height)
        debug        : If True, print detected contour stats

    Returns:
        List of dicts, each with:
            {
                'x': int,      # top-left x
                'y': int,      # top-left y
                'w': int,      # width
                'h': int,      # height
                'cx': int,     # center x
                'cy': int,     # center y
            }
        Sorted top-to-bottom by 'cy'.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold: black borders appear darker than the white paper.
    # Using adaptive threshold handles uneven lighting from camera angle.
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15,
        C=6
    )

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        # Approximate the contour to a polygon
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

        # We want quadrilaterals (4 corners = rectangle)
        if len(approx) != 4:
            continue

        x, y, w, h = cv2.boundingRect(approx)

        # Size filter
        if w < min_size or h < min_size:
            continue
        if w > max_size or h > max_size:
            continue

        # Aspect ratio filter (near-square)
        if approx_square:
            ratio = w / h if h > 0 else 0
            if not (aspect_tolerance < ratio < (1.0 / aspect_tolerance)):
                continue

        if debug:
            print(f"  Box candidate: x={x} y={y} w={w} h={h} ratio={w/h:.2f}")

        boxes.append({
            'x': x, 'y': y,
            'w': w, 'h': h,
            'cx': x + w // 2,
            'cy': y + h // 2,
        })

    # Deduplicate: remove boxes that are nearly identical (within 15px of each other)
    filtered = []
    for box in boxes:
        is_dup = False
        for existing in filtered:
            if abs(box['cx'] - existing['cx']) < 15 and abs(box['cy'] - existing['cy']) < 15:
                is_dup = True
                break
        if not is_dup:
            filtered.append(box)

    # Sort top-to-bottom
    filtered.sort(key=lambda b: b['cy'])

    return filtered


def find_nearest_box_below(question_y: int, boxes: list, tolerance: int = 10) -> dict | None:
    """
    Given a question's Y coordinate (from OCR), find the nearest marking box
    that is BELOW (or at) that Y position.

    Args:
        question_y : Y pixel coordinate of the question text from OCR
        boxes      : List of box dicts from detect_marking_boxes()
        tolerance  : Number of pixels above question_y still accepted (handles OCR imprecision)

    Returns:
        The nearest matching box dict, or None if not found.
    """
    candidates = [b for b in boxes if b['cy'] >= question_y - tolerance]
    if not candidates:
        return None
    return min(candidates, key=lambda b: b['cy'] - question_y)


def draw_detected_boxes(image, boxes: list, color=(0, 255, 0), thickness=2):
    """
    Draw green rectangles and center dots on all detected boxes.
    Returns annotated image (does NOT modify original).

    Args:
        image   : OpenCV BGR image
        boxes   : List of box dicts from detect_marking_boxes()
        color   : BGR color for the overlay rectangle
        thickness: Line thickness

    Returns:
        Annotated copy of the image.
    """
    out = image.copy()
    for i, box in enumerate(boxes):
        x, y, w, h = box['x'], box['y'], box['w'], box['h']
        cx, cy = box['cx'], box['cy']

        # Draw bounding rectangle
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)

        # Draw center dot
        cv2.circle(out, (cx, cy), 5, (0, 0, 255), -1)

        # Label with index and size
        label = f"#{i+1} ({w}x{h})"
        cv2.putText(out, label, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Summary count
    cv2.putText(out, f"Detected: {len(boxes)} boxes", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    return out
