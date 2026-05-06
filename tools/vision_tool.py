"""
tools/vision_tool.py

Analyzes disaster images using Ollama LLaVA 7B running locally.
Requires: ollama pull llava (one-time setup, ~4.7 GB)
No API key. No cloud cost. Runs 100% on localhost.

Fix log:
  - ADDED: Structured VLM output parser — forces SUMMARY/DAMAGE/ACCESS/FULL_DESC format
  - IMPROVED: Graceful fallback when Ollama is not running (informative error, not crash)
  - IMPROVED: Handles both .png and .jpg files; sorted by most recent modification
"""
import base64
import json
import os
import requests
from langchain_core.tools import tool

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_IMAGE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "imagery", "samples"
)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
STRUCTURED_PROMPT = (
    "You are an expert aerial disaster analyst reviewing satellite or drone imagery. "
    "Analyze this image and respond ONLY in this exact format — no other text:\n\n"
    "SUMMARY: <one sentence: overall severity and dominant hazard>\n"
    "DAMAGE: <specific infrastructure damage visible — buildings, roads, bridges>\n"
    "ACCESS: <can emergency vehicles reach this area? what is blocked?>\n"
    "FULL_DESC: <2-3 factual sentences with complete description of what is visible>\n\n"
    "Be specific. Only describe what is directly observable in the image."
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _encode_image(path: str) -> str:
    """Base64-encodes an image file for Ollama API."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _parse_structured_output(raw: str) -> str:
    """
    Parses LLaVA's structured response into a clean formatted string.
    If the model didn't follow the format, returns the raw text wrapped in FULL_DESC.
    """
    lines = {}
    for key in ("SUMMARY", "DAMAGE", "ACCESS", "FULL_DESC"):
        tag = f"{key}:"
        if tag in raw:
            start = raw.index(tag) + len(tag)
            # Find next tag or end of string
            next_tags = [raw.index(f"{k}:") for k in ("SUMMARY", "DAMAGE", "ACCESS", "FULL_DESC")
                         if f"{k}:" in raw and raw.index(f"{k}:") > start]
            end = min(next_tags) if next_tags else len(raw)
            lines[key] = raw[start:end].strip()

    if not lines:
        # Model didn't follow format — wrap raw output
        return f"SUMMARY: Image analyzed\nFULL_DESC: {raw.strip()[:500]}"

    result_parts = []
    if "SUMMARY" in lines:
        result_parts.append(f"  SUMMARY  : {lines['SUMMARY']}")
    if "DAMAGE" in lines:
        result_parts.append(f"  DAMAGE   : {lines['DAMAGE']}")
    if "ACCESS" in lines:
        result_parts.append(f"  ACCESS   : {lines['ACCESS']}")
    if "FULL_DESC" in lines:
        result_parts.append(f"  FULL_DESC: {lines['FULL_DESC']}")

    return "\n".join(result_parts)


def _check_ollama_available() -> bool:
    """Quick connectivity check to Ollama."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Tool 6: Vision Analysis
# ---------------------------------------------------------------------------
@tool
def analyze_disaster_image(image_filename: str = "latest") -> str:
    """
    Analyzes a real disaster satellite/drone image using Ollama LLaVA (local, free).
    Pass a specific filename from the samples directory (e.g. "flood_001.png"),
    or pass "latest" to automatically select the most recently added image.
    Returns a structured damage assessment with SUMMARY, DAMAGE, ACCESS status.
    Requires Ollama to be running locally: start with 'ollama serve'.
    """
    # Check Ollama availability first
    if not _check_ollama_available():
        return (
            "[VISION TOOL — Offline]\n"
            "  Ollama is not running. To enable real image analysis:\n"
            "  1. Download Ollama: https://ollama.ai\n"
            "  2. Run: ollama serve\n"
            "  3. Run: ollama pull llava\n"
            "  Use search_social_reports() for text-based imagery descriptions instead."
        )

    try:
        # Resolve image path
        if image_filename == "latest":
            if not os.path.exists(DEFAULT_IMAGE_DIR):
                return (
                    "[VISION] Image directory not found.\n"
                    "  Run: python ingestion/extract_xview2_subset.py"
                )
            files = [
                f for f in os.listdir(DEFAULT_IMAGE_DIR)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            if not files:
                return (
                    "[VISION] No images in samples directory.\n"
                    "  Run: python ingestion/extract_xview2_subset.py"
                )
            files.sort(
                key=lambda f: os.path.getmtime(os.path.join(DEFAULT_IMAGE_DIR, f)),
                reverse=True
            )
            image_filename = files[0]

        image_path = os.path.join(DEFAULT_IMAGE_DIR, image_filename)
        if not os.path.exists(image_path):
            return (
                f"[VISION] Image not found: {image_filename}\n"
                f"  Available images are in: {DEFAULT_IMAGE_DIR}"
            )

        encoded = _encode_image(image_path)

        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": "llava",
                "prompt": STRUCTURED_PROMPT,
                "images": [encoded],
                "stream": False
            },
            timeout=120
        )
        resp.raise_for_status()
        raw_description = resp.json().get("response", "No response from model.")
        structured = _parse_structured_output(raw_description)

        return (
            f"[OLLAMA LLAVA — {image_filename}]\n"
            f"{structured}"
        )

    except requests.ConnectionError:
        return (
            "[VISION TOOL] Ollama connection lost mid-request.\n"
            "  Ensure Ollama is running: ollama serve"
        )
    except Exception as e:
        return f"[VISION TOOL ERROR] {type(e).__name__}: {e}"
