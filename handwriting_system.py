
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
        
        # ANCHOR CORNER POINTS: Two tiny 1px lines at opposite corners.
        # This forces vpype to see the FULL bounding box for correct scaling.
        # Unlike a full rectangle, these produce nearly zero machine movement
        # when drawn (pen travels <0.1mm). Much safer than a large rect.
        dot = 1  # 1px - scales to ~0.05mm on page, machine won't even register it
        dwg.add(dwg.line((0, 0), (dot, dot), stroke="black", stroke_width=dot))
        dwg.add(dwg.line((source_width - dot, source_height - dot),
                         (source_width, source_height), stroke="black", stroke_width=dot))
        
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

    def convert_to_gcode(self, svg_path: str, target_width_mm: int = 210, target_height_mm: int = 297, scale_divider: float = 3.0) -> Optional[str]:
        """
        Converts SVG to G-Code using vpype.
        Returns the path to the generated gcode file.
        target_width_mm, target_height_mm: Physical dimensions to scale the SVG to.
        scale_divider: Divides all final coordinates by this number to compensate
                       for machine step calibration errors during prototyping.
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
            layout_dim = f"{target_width_mm}mmx{target_height_mm}mm"
            
            cmd = [
                "vpype",
                "read", svg_path,
                # Fit the content into the target physical size
                "scaleto", dim_str_w, dim_str_h,
                # Align to Top-Left. Now that vpype sees the full bounding box
                # (via the anchor rect), this places it exactly at (0,0)
                "layout", "--align", "left", "--valign", "top", f"{target_width_mm}mmx{target_height_mm}mm",
                "linemerge",
                "gwrite",
                "--profile", "gcodemm",
                gcode_path
            ]
            
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Post-process: normalize coordinates so content starts at (0,0) and applying scaling
            self._normalize_gcode(gcode_path, scale_divider)
            
            print(f"[Handwriting] Generated G-Code: {gcode_path}")
            return gcode_path
        except subprocess.CalledProcessError as e:
            print(f"[Handwriting] Error converting to G-Code: {e}")
            if e.stderr:
                print(f"[Handwriting] vpype stderr: {e.stderr.decode()}")
            return None

    def _normalize_gcode(self, gcode_path: str, scale_divider: float = 3.0):
        """
        Post-processes a G-code file to:
        1. STRIP the first connected path (anchor rect/dots - not real marks)
        2. Shift all coordinates so content starts at (0, 0)
        3. Scales the output by dividing by `scale_divider` to fix machine step errors.
        4. Injects Z-axis Pen Up / Pen Down commands between rapid and feed moves.
        """
        import re
        coord_pattern = re.compile(r'([XY])(-?[\d.]+)')
        
        with open(gcode_path, 'r') as f:
            lines = f.readlines()
        
        # --- Step 1: Strip the first connected path (anchor geometry) ---
        first_g00_idx = None
        second_g00_idx = None
        
        for i, line in enumerate(lines):
            stripped = line.strip().upper()
            if stripped.startswith('G00') or stripped.startswith('G0 '):
                if first_g00_idx is None:
                    first_g00_idx = i
                elif second_g00_idx is None:
                    second_g00_idx = i
                    break
        
        filtered_lines = []
        for i, line in enumerate(lines):
            if first_g00_idx is not None and second_g00_idx is not None:
                if first_g00_idx <= i < second_g00_idx:
                    continue
            filtered_lines.append(line)
        
        # --- Step 2: Normalize coordinates to (0, 0) origin ---
        min_x = float('inf')
        min_y = float('inf')
        
        for line in filtered_lines:
            if line.strip().upper().startswith('G0') or line.strip().upper().startswith('G1'):
                for axis, val in coord_pattern.findall(line):
                    v = float(val)
                    if axis == 'X' and v < min_x:
                        min_x = v
                    if axis == 'Y' and v < min_y:
                        min_y = v
        
        if min_x == float('inf') or min_y == float('inf'):
            with open(gcode_path, 'w') as f:
                f.writelines(filtered_lines)
            return
        
        offset_x = min_x
        offset_y = min_y
        
        def shift_coord(match):
            axis = match.group(1)
            val = float(match.group(2))
            if axis == 'X':
                new_val = (val - offset_x) / scale_divider
                return f'X{new_val:.4f}'
            elif axis == 'Y':
                new_val = (val - offset_y) / scale_divider
                return f'Y{new_val:.4f}'
            return match.group(0)
        
        # grbl-servo firmware: M3 S0 = pen DOWN (min), M3 S1000 = pen UP (max)
        PEN_UP = "M3 S1000"  # Servo at max position (pen lifted)
        PEN_DOWN = "M3 S0"   # Servo at min position (pen on paper)
        
        normalized_lines = []
        is_pen_down = False
        first_move_done = False
        
        for line in filtered_lines:
            stripped = line.strip().upper()
            if stripped.startswith('G00') or stripped.startswith('G0 '):
                # Rapid move - pen MUST be UP
                if is_pen_down or not first_move_done:
                    normalized_lines.append(PEN_UP + "\n")
                    is_pen_down = False
                    first_move_done = True
                
                line = coord_pattern.sub(shift_coord, line)
                normalized_lines.append(line)
                
            elif stripped.startswith('G01') or stripped.startswith('G1 '):
                # Drawing move - pen MUST be DOWN
                if not is_pen_down:
                    normalized_lines.append(PEN_DOWN + "\n")
                    is_pen_down = True
                    first_move_done = True
                    
                line = coord_pattern.sub(shift_coord, line)
                normalized_lines.append(line)
                
            else:
                normalized_lines.append(line)
        
        # Always end with the pen up
        if is_pen_down:
            normalized_lines.append(PEN_UP + "\n")
        
        with open(gcode_path, 'w') as f:
            f.writelines(normalized_lines)

# Singleton or utility usage
handwriting = HandwritingSystem()
