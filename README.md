# AI-Powered Automated Copy Checking System with Hardware Integration

An intelligent document grading system that uses dual-camera capture, **GPT-4o Vision** for evaluation, **PaddleOCR** for annotation, and **Arduino-based hardware** for automated page flipping.

## 🚀 Key Features

*   **Dual Camera Capture**: Simultaneously captures from two cameras (Cam 1 & Cam 2).
*   **Parallel Processing**: Evaluates both images **simultaneously** using multi-threading for maximum speed.
*   **Hardware Automation**:
    *   Communicates with Arduino via Serial (default `COM4`).
    *   Sends `flip` signal after processing.
    *   Waits for `capture` signal to continue the loop.
*   **AI Grading**:
    *   **GPT-4o Vision**: Identifies questions and evaluates student answers.
    *   **PaddleOCR**: Locates text and marks "Correct (✓)" or "Wrong (✗)" directly on the image.
*   **Smart Rotation**: Automatically corrects camera mounting orientation (`Cam1: -90°`, `Cam2: +90°`).

## 🛠️ Prerequisites

*   Python 3.10+
*   **OpenAI API Key**
*   **Arduino** (Connected via USB) with page-flipping firmware.
*   Two Webcams.

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

### 1. Start Automated System (Main Production Mode)
This is the main script that runs the full **Capture -> Grade -> Flip -> Repeat** loop.

```bash
python automated_grading.py
```
*   **Note**: Ensure Arduino is connected to `COM4`. To change the port, edit `automated_grading.py`.

### 2. Testing Tools

#### Simple GPT Evaluator for Single Image
Quickly test if GPT can read/grade a specific image.
```bash
python simple_gpt_evaluator.py "path/to/image.png"
```

#### Batch Process Scanned Images
Grade a folder of existing images (no cameras/hardware needed).
```bash
python scanner_processor.py "path/to/folder"
```

#### Manual Camera & OCR Testing
Use `main.py` for component testing without hardware automation.
```bash
# Test Camera Capture only
python main.py camera

# Test Text Extraction (Layout Debugging)
python main.py extract "path/to/image.png"

# Test Preprocessing
python main.py preprocess "path/to/image.png" --method aggressive --output "debug.png"
```

## 📂 Project Structure

*   **`automated_grading.py`**: **MAIN** entry point. Orchestrates cameras, threads, and hardware.
*   **`hardware_interface.py`**: Handles Serial communication with Arduino (`flip`/`capture`).
*   **`camera_system.py`**: Handles camera logic, document detection, cropping, and rotation.
*   **`ocr_engine.py`**: Core grading logic (GPT-4o + PaddleOCR).
*   **`main.py`**: Legacy/Testing hub for individual components.

## ⚙️ Configuration
*   **Camera Rotation**:
    *   Cam 1: Rotated -90° (Counter-Clockwise).
    *   Cam 2: Rotated +90° (Clockwise).
*   **Serial Port**: Default `COM4`, 9600 baud. Change in `automated_grading.py`.

## 📌 Troubleshooting
*   **"Could not connect to COM4"**: Check USB connection or update `COM_PORT` in `automated_grading.py`.
*   **"Cameras not initialized"**: Ensure cameras are plugged in and not used by another app.
