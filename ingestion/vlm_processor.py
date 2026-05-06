"""
ingestion/vlm_processor.py

Real disaster image processing using Ollama LLaVA (local, free).
Processes real xView2 satellite images — NO keyword fallback, NO synthetic data.
Requires: ollama pull llava (one-time, ~4.7 GB)

Fix log:
  - FIXED: Hardcoded "2013-09-12T..." timestamp replaced with real current datetime
"""
import os
import json
import base64
import requests
from datetime import datetime, timezone

OLLAMA_URL = "http://localhost:11434/api/generate"

DISASTER_PROMPT = (
    "You are an expert aerial disaster analyst reviewing satellite/drone imagery. "
    "Respond in exactly this format — no other text:\n\n"
    "SUMMARY: <one sentence: overall severity and dominant hazard>\n"
    "DAMAGE: <specific infrastructure damage — buildings, roads, bridges>\n"
    "ACCESS: <can emergency vehicles reach this area? what is blocked?>\n"
    "FULL_DESC: <2-3 factual sentences with complete description>\n\n"
    "Only describe what you can directly observe in the image."
)


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyze_image_with_llava(image_path: str) -> str | None:
    """
    Sends a real image to Ollama LLaVA running locally.
    Returns the model's structured description or None if Ollama is unavailable.
    """
    try:
        payload = {
            "model":  "llava",
            "prompt": DISASTER_PROMPT,
            "images": [encode_image(image_path)],
            "stream": False
        }
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.ConnectionError:
        print("[VLM] Ollama not running. Start with: ollama serve")
        return None
    except Exception as e:
        print(f"[VLM] Error processing {os.path.basename(image_path)}: {e}")
        return None


def process_images_with_vlm(
    input_dir: str,
    output_json: str,
    max_images: int = 50
) -> None:
    """
    Processes real xView2 disaster images through Ollama LLaVA.
    Writes structured descriptions to output_json for RAG ingestion.
    Refuses to write if Ollama is unavailable — no synthetic fallback.

    Args:
        input_dir: Directory containing extracted xView2 images.
        output_json: Path to write the VLM output JSON.
        max_images: Maximum number of images to process.
    """
    if not os.path.exists(input_dir):
        print(f"[VLM] Image directory not found: {input_dir}")
        print("[VLM] Run ingestion/extract_xview2_subset.py first.")
        return

    images = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])[:max_images]

    if not images:
        print("[VLM] No images found to process.")
        return

    # Test Ollama before starting batch
    print(f"[VLM] Testing Ollama connectivity on first image...")
    test = analyze_image_with_llava(os.path.join(input_dir, images[0]))
    if test is None:
        print(
            "[VLM] FATAL: Ollama unavailable. Cannot process real images.\n"
            "[VLM] Steps:\n"
            "  1. Download Ollama: https://ollama.ai\n"
            "  2. Run: ollama serve\n"
            "  3. Run: ollama pull llava\n"
            "[VLM] Refusing to write synthetic descriptions. Exiting."
        )
        return

    print(f"[VLM] Processing {len(images)} real disaster images with Ollama LLaVA...\n")
    results = []
    process_start = datetime.now(timezone.utc)

    for idx, img_file in enumerate(images):
        img_path = os.path.join(input_dir, img_file)
        print(f"  [{idx+1}/{len(images)}] {img_file}...")

        # Reuse test result for first image
        description = test if idx == 0 else analyze_image_with_llava(img_path)
        test = None  # Only reuse once

        if not description:
            print(f"    [SKIP] No description returned.")
            continue

        # Infer disaster type from xView2 filename conventions
        disaster_type = "Unknown"
        for keyword in ["flood", "hurricane", "earthquake", "volcano", "wildfire", "tsunami", "fire"]:
            if keyword in img_file.lower():
                disaster_type = keyword.capitalize()
                break

        # FIX: Use real current timestamp, not hardcoded "2013-09-12"
        image_timestamp = datetime.now(timezone.utc).isoformat()

        results.append({
            "drone_id":     f"xview2_{idx:04d}",
            "timestamp":    image_timestamp,
            "location":     f"xView2 Disaster Zone — {disaster_type}",
            "observations": description,
            "modality":     "imagery_description",
            "source_file":  img_file,
            "disaster_type": disaster_type,
        })

    if results:
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        elapsed = (datetime.now(timezone.utc) - process_start).seconds
        print(f"\n[VLM] Wrote {len(results)} real VLM descriptions to: {output_json}")
        print(f"[VLM] Total processing time: {elapsed // 60}m {elapsed % 60}s")
    else:
        print("[VLM] No descriptions generated.")


if __name__ == "__main__":
    in_dir   = os.path.join("data", "imagery", "samples")
    out_file = os.path.join("data", "imagery", "real_drone_metadata.json")
    process_images_with_vlm(in_dir, out_file)
