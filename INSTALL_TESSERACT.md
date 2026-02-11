# Installing Tesseract 5 for Windows

## Why Tesseract 5?
✅ **95-98% accuracy** on printed text (best free OCR)  
✅ **Industry standard** - used by Google, Archive.org  
✅ **Perfect for worksheets** and document scanning  
✅ **Completely free** and open source  

---

## Installation Steps

### 1. Download Tesseract 5 Installer

**Download Link**: https://github.com/UB-Mannheim/tesseract/wiki

Or direct download:
- **64-bit Windows**: https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-5.5.0.20241111.exe
- **32-bit Windows**: https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w32-setup-5.5.0.20241111.exe

### 2. Run the Installer

1. Double-click the downloaded `.exe` file
2. Click **"Next"** through the installer
3. **IMPORTANT**: Use the default installation path:
   ```
   C:\Program Files\Tesseract-OCR
   ```
4. Select **"English"** language data (already selected by default)
5. Optional: Select additional languages if needed
6. Click **"Install"**
7. Click **"Finish"**

### 3. Verify Installation

Open PowerShell and run:
```powershell
cd "C:\Program Files\Tesseract-OCR"
.\tesseract.exe --version
```

You should see output like:
```
tesseract 5.5.0
 leptonica-1.85.0
  libgif 5.2.2 : libjpeg 8d (libjpeg-turbo 3.0.4) : libpng 1.6.44 : libtiff 4.7.0 : zlib 1.3.1 : libwebp 1.4.0
```

### 4. Test with Python

Run our test script:
```powershell
cd f:\Mohd_Ahtesham\Projects\ML-Projects\Copy_Checking
.\venv\Scripts\Activate.ps1
python test_ocr.py
```

You should see:
```
✓ Found Tesseract at: C:\Program Files\Tesseract-OCR\tesseract.exe
✓ Tesseract version: 5.5.0
✅ Tesseract is ready to use!
```

### 5. Test OCR on an Image

```powershell
python python_ocr.py --image ocr-test.png --text "your search text"
```

---

## Troubleshooting

### ❌ Error: "Tesseract not found"

**Solution 1**: Verify installation path
```powershell
dir "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

**Solution 2**: If installed elsewhere, update the path in `python_ocr.py`:
```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Your\Custom\Path\tesseract.exe'
```

**Solution 3**: Add to Windows PATH
1. Press `Win + X` → System
2. Advanced system settings → Environment Variables
3. Under System variables, find **Path** → Edit
4. Click **New** → Add: `C:\Program Files\Tesseract-OCR`
5. Click **OK** and restart PowerShell

### ❌ Error: "Missing dependencies"

Install required Python packages:
```powershell
.\venv\Scripts\Activate.ps1
pip install pytesseract pillow opencv-python
```

### 🐌 OCR is slow

This is normal for high-quality OCR. For faster processing:
- Use lower resolution images
- Crop to text areas only
- Consider GPU acceleration (requires CUDA setup)

---

## Advanced Configuration

### Maximum Quality Settings

In `python_ocr.py`, the OCR is configured with:
```python
config = r'--oem 1 --psm 3'
```

Where:
- `--oem 1`: Use LSTM neural network (best accuracy)
- `--psm 3`: Automatic page segmentation

### Other PSM Modes

If you have specific document types:
```python
--psm 6  # Assume a single uniform block of text
--psm 11 # Sparse text. Find as much text as possible
--psm 13 # Raw line (for single text lines)
```

---

## Performance Comparison

| OCR Engine | Accuracy | Speed | Setup |
|------------|----------|-------|-------|
| **Tesseract 5** | ⭐⭐⭐⭐⭐ 95-98% | Medium | Easy |
| EasyOCR | ⭐⭐⭐⭐ 85-92% | Slow | Easy |
| PaddleOCR | ⭐⭐⭐⭐⭐ 96%+ | Fast | Complex |
| Google Vision | ⭐⭐⭐⭐⭐+ 99%+ | Fast | API Key |

**Tesseract 5 is the best balance of accuracy and ease of use!**

---

## Success! 🎉

Once installed, your OCR system will have:
- ✅ 95-98% accuracy on printed documents
- ✅ Perfect for worksheet grading
- ✅ No API costs
- ✅ Works offline
- ✅ Industry-standard quality

Happy OCR-ing! 📄✨
