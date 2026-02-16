
import os
import sys
import argparse
import json
from dotenv import load_dotenv

# Import our modular engines
from ocr_engine import evaluate_with_gpt

# Load environment variables
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Simple GPT Evaluator")
    parser.add_argument("image", help="Path to input image")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set in .env")
        return

    print(f"Evaluating: {args.image}...")
    
    try:
        # Call the modular function
        result = evaluate_with_gpt(
            image_path=args.image,
            api_key=api_key,
            preprocess=True 
        )
        
        # Print result nicely
        print("\n" + "="*50)
        print("GPT EVALUATION RESULT")
        print("="*50)
        print(json.dumps(result, indent=2))
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
