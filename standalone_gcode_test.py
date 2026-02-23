import os
import cv2
from ocr_engine import locate_and_annotate
from handwriting_system import HandwritingSystem

# =========================================================================
# STANDALONE G-CODE GENERATOR
# Takes two existing images, generates dummy "correct" marks,
# merges them perfectly, and outputs a G-code file layout out for A4 side-by-side.
# =========================================================================

def generate_test_gcode(left_image_path: str, right_image_path: str):
    """Generates G-code directly from two images without needing camera/OpenAI."""
    
    print("\n[1] Starting G-Code Generation Test")
    if not os.path.exists(left_image_path) or not os.path.exists(right_image_path):
        print("ERROR: Please provide two valid image paths.")
        return

    # Creating a dummy GPT response where every question on the page is marked Correct
    # This just tells the OCR engine "go find these lines of text and put a tick next to them"
    dummy_gpt_response = {
        "Q1": {"question": "What is the formula", "isAnswerCorrect": True},
        "Q2": {"question": "Find the perimeter", "isAnswerCorrect": True},
        "Q3": {"question": "What is the value", "isAnswerCorrect": True},
        "Q4": {"question": "Write the formula for the", "isAnswerCorrect": False}, 
        "Q5": {"question": "Simplify", "isAnswerCorrect": True},
        "Q6": {"question": "What is a right angle", "isAnswerCorrect": True},
        "Q7": {"question": "Find the area of a square", "isAnswerCorrect": False},
        "Q8": {"question": "Convert 1 meter into", "isAnswerCorrect": True},
        "Q9": {"question": "What is the sum", "isAnswerCorrect": True}
    }

    try:
        # Step 2: Annotate Left Page
        print("\n[2] Processing Left Page (this runs PaddleOCR to find text coordinates)...")
        left_out, left_annotations = locate_and_annotate(
            left_image_path, 
            dummy_gpt_response, 
            output_path="handwriting_output/standalone_left.jpg"
        )
        print(f"  -> Found {len(left_annotations)} marks for left page.")

        # Step 3: Annotate Right Page
        print("\n[3] Processing Right Page...")
        right_out, right_annotations = locate_and_annotate(
            right_image_path, 
            dummy_gpt_response, 
            output_path="handwriting_output/standalone_right.jpg"
        )
        print(f"  -> Found {len(right_annotations)} marks for right page.")

        # Step 4: Run Handwriting System to Generate G-code
        print("\n[4] Generating Side-by-Side SVG and G-code...")
        
        # We need the original image shapes to scale the SVG correctly
        img_left = cv2.imread(left_image_path)
        img_right = cv2.imread(right_image_path)
        
        # -------------------------------------------------------------
        # IMPORTANT: MERGE ANNOTATIONS LIKE THE MAIN PIPELINE DOES
        # The right page needs to be shifted right by the width of the left page
        # -------------------------------------------------------------
        
        # Calculate scores
        left_correct = sum(1 for a in left_annotations if a['type'] == 'correct')
        right_correct = sum(1 for a in right_annotations if a['type'] == 'correct')
        left_score_str = f"Score: {left_correct}/{len(left_annotations)}" if left_annotations else ""
        right_score_str = f"Score: {right_correct}/{len(right_annotations)}" if right_annotations else ""

        # Construct final coordinate lists
        combined_annotations = []
        
        # Left page: keep coordinates as-is
        for ann in left_annotations:
            combined_annotations.append(ann)

        # Right page: Shift all X coordinates by left_width
        left_width = img_left.shape[1]
        for ann in right_annotations:
            shifted_ann = {
                "type": ann["type"],
                "x": ann["x"] + left_width,
                "y": ann["y"]
            }
            combined_annotations.append(shifted_ann)

        # Total dimensions for the single merged SVG
        total_width = left_width + img_right.shape[1]
        max_height = max(img_left.shape[0], img_right.shape[0])

        # Initialize the hardware wrapper (HandwritingSystem)
        hw_system = HandwritingSystem()
        
        # Generate ONE single SVG covering both widths
        svg_path = hw_system.generate_svg(
            filename="standalone_combined.svg",
            annotations=combined_annotations,
            feedback_text=f"{left_score_str}   |   {right_score_str}",
            source_width=total_width,
            source_height=max_height
        )
        if not svg_path:
            print("ERROR: SVG generation failed.")
            return

        # Target dimensions for two A4 sheets side-by-side: 420mm (width) x 297mm (height)
        final_gcode_path = hw_system.convert_to_gcode(
            svg_path,
            target_width_mm=420,
            target_height_mm=297
        )

        if final_gcode_path:
            print(f"\n✅ SUCCESS! Test G-code generated at: {final_gcode_path}")
            print("You can now run `python machine_movement.py` to test it.")
            
            # Just to make life easier, point machine_movement.py to this new file programatically:
            print("\nNote: Make sure GCODE_FILE in machine_movement.py points to this new file!")

    except Exception as e:
        print(f"\n❌ ERROR during test execution: {e}")


if __name__ == "__main__":
    # Ensure the output directory exists
    os.makedirs("handwriting_output", exist_ok=True)
    
    # Let the user know they need to supply test images
    print("This script takes two source images and outputs combined G-code.")
    left_img = input("Enter path to LEFT page image (e.g. captures/left.jpg): ").strip()
    right_img = input("Enter path to RIGHT page image (e.g. captures/right.jpg): ").strip()
    
    generate_test_gcode(left_img, right_img)
