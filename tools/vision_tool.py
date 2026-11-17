"""
tools/vision_tool.py

Analyzes disaster images using Ollama LLaVA 7B running locally.
Requires: ollama pull llava (one-time setup, ~4.7 GB)
No API key. No cloud cost. Runs 100% on localhost.
"""
import base64
import json
import os
import re
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
    "Analyze this image and respond ONLY in this exact format - no other text:\n\n"
    "SUMMARY: <one sentence: overall severity and dominant hazard>\n"
    "DAMAGE: <specific infrastructure damage visible - buildings, roads, bridges>\n"
    "ACCESS: <can emergency vehicles reach this area? what is blocked?>\n"
    "FULL_DESC: <2-3 factual sentences with complete description of what is visible>\n\n"
    "Be specific. Only describe what is directly observable in the image."
)

# Classification mode: the benchmark needs a 4-class damage label, which a
# free-form scene description cannot provide. This prompt is used only when
# task="classify"; the production describe path is unchanged.
DAMAGE_CLASSES = ("no-damage", "minor-damage", "major-damage", "destroyed")
CLASSIFY_PROMPT = (
    "You are an expert satellite disaster damage assessor. Classify the building "
    "in this image crop into exactly one damage class.\n\n"
    "Classes:\n"
    "  no-damage      - building intact, no visible structural damage\n"
    "  minor-damage   - slight or partial damage, structure largely intact\n"
    "  major-damage   - severe structural damage, roof/walls substantially compromised\n"
    "  destroyed      - building collapsed, reduced to rubble, or not recognizable\n\n"
    "Reply with EXACTLY one line and nothing else:\n"
    "DAMAGE_CLASS: <no-damage|minor-damage|major-damage|destroyed>"
)


def _parse_classification(raw: str) -> str:
    """Extract the damage class; return UNCLASSIFIED when the model ignores the format."""
    m = re.search(r"DAMAGE_CLASS\s*:\s*(no-damage|minor-damage|major-damage|destroyed)",
                  raw or "", re.IGNORECASE)
    if m:
        return m.group(1).lower()
    return "UNCLASSIFIED"


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
        # Model didn't follow format - wrap raw output
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


def _get_available_vision_model() -> str:
    """Finds the best available local vision model with graceful fallback."""
    preferred = os.getenv("OLLAMA_VISION_MODEL", "qwen3-vl:4b").strip()
    fallbacks = ["qwen3-vl:4b", "qwen2.5vl:3b", "llava:latest", "llava"]
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        if resp.status_code == 200:
            installed = [m.get("name", "") for m in resp.json().get("models", [])]
            if preferred in installed:
                return preferred
            for fb in fallbacks:
                for inst in installed:
                    if fb in inst:
                        return inst
    except Exception:
        pass
    return preferred


# ---------------------------------------------------------------------------
# Tool 6: Vision Analysis
# ---------------------------------------------------------------------------
@tool
def analyze_disaster_image(image_filename: str = "latest", task: str = "describe") -> str:
    """
    Analyzes a real disaster satellite/drone image using local Ollama Vision AI.
    task="describe" (default): structured SUMMARY/DAMAGE/ACCESS scene assessment.
    task="classify": single 4-class building damage label (for evaluation only).
    Pass a specific filename from the samples directory, or "latest" for newest.
    Requires Ollama running locally.
    """
    # Check Ollama availability first
    if not _check_ollama_available():
        return (
            "[VISION TOOL - Offline]\n"
            "  Ollama is not running. To enable real image analysis:\n"
            "  1. Ensure Ollama is running at http://localhost:11434\n"
            "  2. Pull a vision model: ollama pull qwen3-vl:4b\n"
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
        vision_model = _get_available_vision_model()
        is_classify = task.lower() == "classify"
        prompt = CLASSIFY_PROMPT if is_classify else STRUCTURED_PROMPT

        # keep_alive=0 in production releases the 3.3-4.7 GB model from RAM after
        # each answer. Benchmarks run many sequential calls; OLLAMA_VISION_KEEP_ALIVE
        # lets them hold the model warm without changing the production default.
        keep_alive = os.getenv("OLLAMA_VISION_KEEP_ALIVE", "0")

        # Call Ollama; default keep_alive=0 immediately releases RAM after inference
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": vision_model,
                "prompt": prompt,
                "images": [encoded],
                "stream": False,
                "keep_alive": keep_alive,
            },
            # Local vision on a 4 GB GPU can exceed 2 minutes per image under
            # load; a 120 s timeout silently converted answers into errors.
            timeout=600
        )
        resp.raise_for_status()
        raw_description = resp.json().get("response", "No response from model.")
        if is_classify:
            label = _parse_classification(raw_description)
            return f"[OLLAMA VISION CLASS ({vision_model}) - {image_filename}]\nDAMAGE_CLASS: {label}"
        structured = _parse_structured_output(raw_description)

        return (
            f"[OLLAMA VISION ({vision_model}) - {image_filename}]\n"
            f"{structured}"
        )

    except requests.ConnectionError:
        return (
            "[VISION TOOL] Ollama connection lost mid-request.\n"
            "  Ensure Ollama is running: ollama serve"
        )
    except Exception as e:
        return f"[VISION TOOL ERROR] {type(e).__name__}: {e}"
