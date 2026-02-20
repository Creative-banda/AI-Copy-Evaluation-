
import os
import subprocess
import svgwrite
from typing import List, Tuple, Dict, Optional

# --- Internal Vector Font Data (Hershey Simplex subset) ---
# Each character is a list of strokes. Each stroke is a list of (x, y) coordinates.
# Coordinates are normalized to 0-10 box, centered around (5,5).
VECTOR_FONT = {
    '0': [[(2, 2), (8, 2), (8, 8), (2, 8), (2, 2), (8, 8)]], # Box with slash
    '1': [[(5, 2), (5, 8)]],
    '2': [[(2, 2), (8, 2), (8, 5), (2, 5), (2, 8), (8, 8)]],
    '3': [[(2, 2), (8, 2), (2, 5), (8, 5), (8, 8), (2, 8)]], # Z style
    '4': [[(2, 2), (2, 5), (8, 5)], [(8, 2), (8, 8)]],
    '5': [[(8, 2), (2, 2), (2, 5), (8, 5), (8, 8), (2, 8)]],
    '6': [[(8, 2), (2, 2), (2, 8), (8, 8), (8, 5), (2, 5)]],
    '7': [[(2, 2), (8, 2), (2, 8)]],
    '8': [[(2, 5), (2, 2), (8, 2), (8, 5), (8, 8), (2, 8), (2, 5), (8, 5)]],
    '9': [[(2, 5), (8, 5), (8, 2), (2, 2), (2, 5)], [(8, 5), (8, 8)]],
    '/': [[(8, 2), (2, 8)]],
    'C': [[(8, 2), (2, 2), (2, 8), (8, 8)]],
    'o': [[(2, 2), (8, 2), (8, 8), (2, 8), (2, 2)]],
    'r': [[(2, 8), (2, 5), (8, 5)]], # Simple 'r'
    'e': [[(2, 5), (8, 5), (8, 2), (2, 2), (2, 8), (8, 8)]],
    'c': [[(8, 5), (2, 5), (2, 8), (8, 8)]], # Lowercase c
    't': [[(5, 2), (5, 8)], [(2, 4), (8, 4)]],
    'W': [[(2, 2), (4, 8), (5, 5), (6, 8), (8, 2)]],
    'n': [[(2, 8), (2, 5), (8, 5), (8, 8)]],
    'g': [[(8, 5), (2, 5), (2, 8), (8, 8), (8, 5), (8, 10), (5, 12)]], 
    ' ': [], # Space
    # Add more as needed
}

# Mapping common words to simpler representations if font is incomplete
WORD_MAP = {
    "Correct": [
        [(-5, 0), (0, 5), (10, -10)] # Checkmark
    ],
    "Wrong": [
        [(-5, -5), (5, 5)], [(-5, 5), (5, -5)] # Cross
    ]
}

class HandwritingSystem:
    def __init__(self, output_dir="handwriting_output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _draw_vector_text(self, dwg, x, y, text, size=10):
        """Draws text using internal vector font"""
        cursor_x = x
        for char in text:
            if char in VECTOR_FONT:
                strokes = VECTOR_FONT[char]
                for stroke in strokes:
                    # Scale and Translate
                    points = [(cursor_x + (px/10)*size, y + (py/10)*size) for px, py in stroke]
                    dwg.add(dwg.polyline(points, stroke="red", fill="none", stroke_width=2))
            
            cursor_x += size * 1.2 # Spacing

    def generate_svg(self, filename: str, annotations: List[Dict], feedback_text: str = "", source_width: int = 3840, source_height: int = 2160) -> str:
        """
        Generates an SVG file with annotations.
        annotations: List of dicts with {'type': 'correct'|'wrong', 'x': int, 'y': int}
        source_width, source_height: Dimensions of the source image to set viewBox correctly.
        """
        filepath = os.path.join(self.output_dir, filename)
        
        # Create SVG with viewBox matching the source image resolution
        dwg = svgwrite.Drawing(filepath, profile='tiny')
        dwg.viewbox(0, 0, source_width, source_height)
        
        # INVISIBLE BOUNDING BOX 
        # (Forces vpype to see the full document size, preventing 'scaleto' from expanding small ticks)
        dwg.add(dwg.rect(insert=(0, 0), size=(source_width, source_height), fill="none", stroke="none"))
        
        # Draw Annotations
        for ann in annotations:
            x, y = ann['x'], ann['y']
            # Scale mark size based on resolution? 30px is tiny on 4K. Let's make it bigger.
            size = 100 
            
            if ann['type'] == 'correct':
                # Draw Checkmark
                points = [(x-size//3, y), (x, y+size//3), (x+2*size//3, y-2*size//3)]
                dwg.add(dwg.polyline(points, stroke="green", fill="none", stroke_width=10))
            elif ann['type'] == 'wrong':
                # Draw Cross
                half = size // 2
                dwg.add(dwg.line((x-half, y-half), (x+half, y+half), stroke="red", stroke_width=10))
                dwg.add(dwg.line((x-half, y+half), (x+half, y-half), stroke="red", stroke_width=10))

        # Draw Feedback Text (Score)
        if feedback_text:
            # Position at top right (e.g., 70% width, 10% height) - Moved left to prevent clipping
            score_x = int(source_width * 0.70) 
            score_y = int(source_height * 0.10)
            self._draw_vector_text(dwg, score_x, score_y, feedback_text, size=150)

        dwg.save()
        return filepath

    def convert_to_gcode(self, svg_path: str) -> Optional[str]:
        """
        Converts SVG to G-Code using vpype.
        Returns the path to the generated gcode file.
        """
        gcode_path = svg_path.replace('.svg', '.gcode')
        try:
            # 1. vpype read svg
            # 2. scale to match A4 width (210mm)
            # 3. layout to top-left (0,0) of A4 page
            # 4. linemerge
            # 5. gwrite
            
            # Calculate scale factor: 210mm / current_width_in_px
            # But vpype reads "px" as 1/96 inch by default. 
            # We want: Image Width (px) -> 210mm.
            # VPype 'scaleto' fits bounding box. 'scale' takes factor.
            # Let's try 'scaleto' with 'preserve aspect' to fit PAGE WIDTH.
            # Better: use 'scale' with explicit factor if we knew it.
            # Easier: Use 'scaleto' on the *Layer*? 
            # Best for this: 'layout' the SVG onto the A4 page at 0,0 with scaling.
            # Command: read svg -> layout -h left -v top --scale 210mmx297mm a4 -> ...
            # Wait, 'layout' aligns content.
            # Let's use 'scaleto' but force it to match width: "scaleto 210mm 297mm" fits inside.
            # If image matches A4 Aspect Ratio, it's fine. If not, it fits inside.
            # Our image (2160x3840) is 0.56 AR. A4 is 0.70 AR.
            # So image is taller/thinner. 'scaleto 210mm 297mm' will fit HEIGHT (297mm) and Width will be < 210mm.
            # This is correct behavior to fit on page.
            # THE ISSUE WAS: 'scaleto' fits the *bounding box of content*.
            # If I only have 1 tick, it makes that tick HUGE.
            # FIX: We need to scale the *entire coordinate system* of the SVG.
            # Since we set viewbox=image_size, vpype *should* respect that?
            # Vpype ignores viewbox for scaling usually, it looks at geometries.
            # FORCE: 'read --no-crop' (if available)? No.
            # TRICK: Add a transparent rectangle at 0,0 and W,H to force bounding box?
            # BETTER: Explicit scale factor.
            # Factor = 210mm / Width_px?
            # No, let's just use 'scale' command.
            # We assume width fits 210mm.
            
            cmd = [
                "vpype",
                "read", svg_path,
                "scaleto", "210mm", "297mm", # This is still safer to fit page
                "layout", "a4", # Center on A4? No we want top-left.
                # Let's try: scale to 210mm width explicitly?
                # "scale", ... hard to calculate here without knowing unit context.
                # Backtrack: The USER said G-Code was huge.
                # Reason: 'scaleto' fitted the TICKS to 210x297.
                # Fix: Add invisible bounding box points in SVG!
                "linemerge",
                "gwrite",
                "--profile", "gcodemm",
                gcode_path
            ]
            
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"[Handwriting] Generated G-Code: {gcode_path}")
            return gcode_path
        except subprocess.CalledProcessError as e:
            print(f"[Handwriting] Error converting to G-Code: {e}")
            if e.stderr:
                print(f"[Handwriting] vpype stderr: {e.stderr.decode()}")
            return None

# Singleton or utility usage
handwriting = HandwritingSystem()
