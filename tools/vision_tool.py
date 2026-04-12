"""
tools/vision_tool.py

Analyzes disaster images using Ollama LLaVA 7B running locally.
Requires: ollama pull llava (one-time setup, ~4.7 GB)
No API key. No cloud cost. Runs 100% on localhost.
"""
import base64
import json
import os
import requests
from langchain_core.tools import tool

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_IMAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "imagery", "samples")

DISASTER_PROMPT = (
    "You are an expert aerial disaster analyst. Analyze this satellite or drone image "
    "and describe in 3 factual sentences: (1) What infrastructure damage is visible? "
    "(2) What natural hazards (flooding, fire, debris) are present? "
    "(3) What is the access status for emergency vehicles? "
    "Be specific and factual. Do not speculate beyond what is visible."
)

def _encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

@tool
def analyze_disaster_image(image_filename: str = "latest") -> str:
    """
    Analyzes a real disaster satellite/drone image using Ollama LLaVA (local, free).
    Pass a filename from the samples directory, or 'latest' for the most recent image.
    Returns a factual damage assessment for emergency responders.
    """
    try:
        # Resolve image path
        if image_filename == "latest":
            files = [f for f in os.listdir(DEFAULT_IMAGE_DIR) if f.endswith((".png", ".jpg"))]
            if not files:
                return "[VISION] No images found in samples directory. Run ingestion/extract_xview2_subset.py first."
            # Pick the most recently modified
            files.sort(key=lambda f: os.path.getmtime(os.path.join(DEFAULT_IMAGE_DIR, f)), reverse=True)
            image_filename = files[0]

        image_path = os.path.join(DEFAULT_IMAGE_DIR, image_filename)
        if not os.path.exists(image_path):
            return f"[VISION] Image not found: {image_path}"

        encoded = _encode_image(image_path)

        payload = {
            "model": "llava",
            "prompt": DISASTER_PROMPT,
            "images": [encoded],
            "stream": False
        }

        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        description = resp.json().get("response", "No response from model.")

        return (
            f"[OLLAMA LLAVA — {image_filename}]\n"
            f"  {description.strip()}"
        )

    except requests.ConnectionError:
        return (
            "[VISION TOOL] Ollama not running. Start it with: ollama serve\n"
            "Then ensure LLaVA is installed: ollama pull llava"
        )
    except Exception as e:
        return f"[VISION TOOL ERROR] {e}"
