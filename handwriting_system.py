
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

    def convert_to_gcode(self, svg_path: str, target_width_mm: int = 210, target_height_mm: int = 297) -> Optional[str]:
        """
        Converts SVG to G-Code using vpype.
        Returns the path to the generated gcode file.
        target_width_mm, target_height_mm: Physical dimensions to scale the SVG to.
        """
        gcode_path = svg_path.replace('.svg', '.gcode')
        try:
            # 1. vpype read svg
            # 2. scale to match target dimensions
            # 3. linemerge
            # 4. gwrite
            
            # Using 'scaleto' with target dimensions + explicit layout alignment
            dim_str_w = f"{target_width_mm}mm"
            dim_str_h = f"{target_height_mm}mm"
            layout_dim = f"{target_width_mm}mmx{target_height_mm}"
            
            cmd = [
                "vpype",
                "read", svg_path,
                # Fit into target physical size
                "scaleto", dim_str_w, dim_str_h, 
                # CRITICAL: Align to Top-Left of the "Page", 
                # ensuring (0,0) in SVG matches (0,0) in G-Code.
                # Without this, "scaleto" might center content, shifting X right.
                "layout", "--halign", "left", "--valign", "top", layout_dim,
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
