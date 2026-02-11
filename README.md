# AI-Powered Automated Copy Checking System

An intelligent document grading system that uses dual-camera capture, **GPT-4o Vision** for evaluation, and **PaddleOCR** for precise location-based annotation.

## 🚀 Key Features

*   **Dual Camera Capture**: Simultaneously captures feeds from two cameras (e.g., student view + worksheet view).
*   **Automated Document Detection**: Automatically detects document contours, crops, and perspective-corrects the image.
*   **AI Grading (GPT-4o)**:
    *   Compresses images efficiently for token optimization (JPEG 95% quality, resized).
    *   Sends the worksheet to **OpenAI GPT-4o** to identify questions and evaluate answers.
*   **Visual Annotation**:
    *   Uses **PaddleOCR** to locate the exact position of questions on the page.
    *   Marks questions as **Correct (✓)** or **Wrong (✗)** directly on the image based on GPT's evaluation.
*   **Layout Preservation Mode**: Debugging tool to extract text while preserving the visual layout.

## 🛠️ Prerequisites

*   Python 3.10+
*   **Tesseract OCR** (Optional, for fallback)
*   **OpenAI API Key**

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Creative-banda/AI-Copy-Evaluation-.git
    cd AI-Copy-Evaluation-
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up Environment Variables:**
    Create a `.env` file in the root directory:
    ```env
    OPENAI_API_KEY=your_openai_api_key_here
    ```

## 🖥️ Usage

### 1. Auto-Grading Mode (Default)
Starts the camera system. When a document is stable for 3 seconds, it is captured, processed, sent to GPT, and annotated automatically.

```bash
python main.py
# OR
python main.py camera --camera1 0 --camera2 1
```

*   **Output**: Saved in `captured_copies/camera_timestamp/`
    *   `original_color.png`: The cropped document.
    *   `gpt_compressed.jpg`: Optimized image sent to AI.
    *   `original_gray_GRADED.png`: Final annotated image with grades.

### 2. Text Extraction Mode (Layout Debugging)
Extracts text from a specific image while attempting to preserve the original visual layout (lines/columns). Useful for testing how the OCR "sees" the document.

```bash
python main.py extract "path/to/image.png"
```

### 3. Manual OCR Testing
Test PaddleOCR detection on a specific image.

```bash
# Find specific text
python main.py ocr "path/to/image.png" "Question 1"

# Extract all text (unstructured)
python ocr_engine.py "path/to/image.png"
```

## 📂 Project Structure

*   **`main.py`**: Entry point. Handles CLI arguments and orchestrates the workflow.
*   **`camera_system.py`**: Manages OpenCV camera feeds, document detection, and image saving.
*   **`ocr_engine.py`**: Wraps PaddleOCR and GPT-4o logic. Handles text detection and JSON parsing.
*   **`captured_copies/`**: Stores all capture sessions.

## ⚙️ Configuration
*   **Image Optimization**: GPT images are resized to max `2048px` and saved at `95%` JPEG quality.
*   **OCR Engine**: Defaults to `PaddleOCR` (v2.9.1) for superior accuracy.

---
**Note**: Ensure your camera IDs (0, 1) are correct if using external webcams.
