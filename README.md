# Copy Checking System

OCR-based worksheet grading and dual-camera document capture system.

## Setup

### 1. Virtual Environment (Recommended)

This project uses a virtual environment to isolate dependencies:

```bash
# Already created - just activate it:
.\venv\Scripts\Activate.ps1   # PowerShell
# OR
venv\Scripts\activate.bat      # Command Prompt
```

### 2. Install Dependencies

Dependencies are already installed, but if you need to reinstall:

```bash
pip install -r requirements.txt
```

## Components

### 1. Tesseract 5 OCR Integration (`python_ocr.py`)

**✨ NOW USING: Tesseract 5 - Industry Standard OCR!**

**Why Tesseract 5?**
- ✅ **95-98% accuracy** on printed documents (best free OCR)
- ✅ **Industry standard** - used by Google, Archive.org
- ✅ **Perfect for worksheets** and educational materials
- ✅ **LSTM neural network** for superior recognition
- ✅ **Completely free** and open source
- ✅ **Fast processing** - optimized C++ engine

**Installation Required:**
1. Install Tesseract 5 binary: See [INSTALL_TESSERACT.md](INSTALL_TESSERACT.md)
2. Python packages installed automatically in venv

#### Usage:

```bash
# Using the virtual environment (recommended):
.\run_ocr.ps1 --evaluate --image test_image.png

# Or activate venv manually:
.\venv\Scripts\Activate.ps1
python python_ocr.py --evaluate --image test_image.png
```


#### Command Line Options:

```bash
# Evaluate and grade a worksheet with OpenAI
python python_ocr.py --evaluate --image worksheet.png --api-key YOUR_API_KEY

# Find specific text in an image
python python_ocr.py --image document.png --text "Question 1"

# Specify output files
python python_ocr.py --evaluate --image test.png --out-image graded.png --out-json results.json
```

### 2. Dual-Camera Capture System (`multiple_camera.py`)

Automatic document detection and high-resolution capture from two cameras.

#### Features:
- Real-time contour detection using Otsu thresholding + Canny edges
- Processes at 640px for speed, captures at maximum camera resolution
- Auto-saves when stable contours detected for 30 frames
- Independent operation of both cameras
- Saves high-quality PNG images (lossless)

#### Usage:

```bash
# Using the virtual environment:
.\run_camera.ps1

# Or activate venv manually:
.\venv\Scripts\Activate.ps1
python multiple_camera.py
```

#### Controls:
- `d` - Toggle debug mode (shows detection steps)
- `+`/`-` - Adjust minimum contour area
- `]`/`[` - Adjust maximum area ratio
- `q` or `ESC` - Quit

#### Output Structure:
```
captured_copies/
├── cam1_contour_20260210_143052.png  # Full image with detection box
├── cam2_contour_20260210_143105.png
└── cropped/
    ├── cam1_cropped_20260210_143052.png  # High-res cropped document
    └── cam2_cropped_20260210_143105.png
```

## Requirements

- Python 3.10+
- **Tesseract 5.x** (binary installation required - see [INSTALL_TESSERACT.md](INSTALL_TESSERACT.md))
- OpenCV
- pytesseract (Python wrapper)
- Pillow (PIL)
- OpenAI API (for grading)
- NumPy

See `requirements.txt` for complete Python package list.

## Environment Variables

```bash
# Set your OpenAI API key:
$env:OPENAI_API_KEY = "your-api-key-here"  # PowerShell
# OR
set OPENAI_API_KEY=your-api-key-here       # Command Prompt
```

## Migration to Tesseract 5

The OCR system now uses **Tesseract 5** for maximum accuracy:

### What Changed:
- ✅ **95-98% accuracy** (vs 85-92% with EasyOCR)
- ✅ LSTM neural network for best-in-class recognition
- ✅ Optimized for printed text (perfect for worksheets)
- ✅ Faster processing than deep learning alternatives
- ✅ Lower memory usage

### Installation:
See **[INSTALL_TESSERACT.md](INSTALL_TESSERACT.md)** for detailed step-by-step instructions.

**Quick Install:**
1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install to default path: `C:\Program Files\Tesseract-OCR`
3. Run `python test_ocr.py` to verify

### Why Tesseract 5 Over Others?

| Feature | Tesseract 5 | EasyOCR | PaddleOCR |
|---------|-------------|---------|-----------|
| **Accuracy (Printed)** | ⭐⭐⭐⭐⭐ 95-98% | ⭐⭐⭐⭐ 85-92% | ⭐⭐⭐⭐⭐ 96%+ |
| **Speed** | ⚡⚡⚡ Fast | ⚡ Slow | ⚡⚡ Medium |
| **Memory Usage** | 💚 Low | ❤️ High | 💛 Medium |
| **Windows Support** | ✅ Excellent | ✅ Good | ❌ Issues |
| **Setup Difficulty** | 🟢 Easy | 🟢 Easy | 🔴 Complex |
| **Best For** | Printed docs | Multilingual | Chinese/Scene text |

**Tesseract 5 is the best choice for worksheet grading!**

## Troubleshooting

### Virtual Environment Not Activating
```bash
# PowerShell execution policy issue:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### PaddleOCR Model Download
First run downloads models automatically. Ensure internet connection.

### Camera Not Found
- Check camera indices in `multiple_camera.py`
- Try different camera IDs (0, 1, 2...)

## Development

To add new dependencies:

```bash
# Activate venv first
.\venv\Scripts\Activate.ps1

# Install package
pip install package-name

# Update requirements.txt
pip freeze > requirements.txt
```

## License

[Add your license here]
