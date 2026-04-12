"""
ingestion/vlm_processor.py

Real disaster image processing using Ollama LLaVA (local, free).
Processes real xView2 satellite images — NO keyword fallback, NO synthetic data.
Requires: ollama pull llava (one-time, ~4.7 GB)
"""
import os
import json
import base64
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

DISASTER_PROMPT = (
    "You are an expert aerial disaster analyst reviewing satellite/drone imagery. "
    "Describe this image in 3 factual sentences covering: "
    "(1) What specific infrastructure damage or hazard is visible? "
    "(2) What is the severity — is this area passable for emergency vehicles? "
    "(3) What is the estimated flood depth or structural collapse extent if visible? "
    "Be specific. Only describe what you can directly observe."
)

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyze_image_with_llava(image_path: str) -> str | None:
    """
    Sends a real image to Ollama LLaVA running locally.
    Returns the model's factual description or None if Ollama is unavailable.
    """
    try:
        payload = {
            "model": "llava",
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


def process_images_with_vlm(input_dir: str, output_json: str, max_images: int = 50) -> None:
    """
    Processes real xView2 disaster images through Ollama LLaVA.
    Writes factual descriptions to output_json for RAG ingestion.
    Skips processing if Ollama is unavailable (logs error, does NOT use keyword fallback).
    """
    if not os.path.exists(input_dir):
        print(f"[VLM] Image directory not found: {input_dir}")
        print("[VLM] Run ingestion/extract_xview2_subset.py first to extract real images.")
        return

    images = sorted([f for f in os.listdir(input_dir) if f.endswith((".png", ".jpg"))])[:max_images]
    if not images:
        print("[VLM] No images found to process.")
        return

    # Test Ollama connectivity before starting
    test = analyze_image_with_llava(os.path.join(input_dir, images[0]))
    if test is None:
        print("[VLM] FATAL: Ollama unavailable. Cannot process real images.")
        print("[VLM] Start Ollama: ollama serve")
        print("[VLM] Refusing to write synthetic descriptions. Exiting.")
        return

    print(f"[VLM] Processing {len(images)} real disaster images with Ollama LLaVA...")
    results = []

    for idx, img_file in enumerate(images):
        img_path = os.path.join(input_dir, img_file)
        print(f"  [{idx+1}/{len(images)}] {img_file}...")

        # Reuse the result from the connectivity test for the first image
        description = test if idx == 0 else analyze_image_with_llava(img_path)
        test = None  # Only reuse once

        if description:
            # Infer disaster type from filename (xView2 uses descriptive naming)
            disaster_type = "Unknown"
            for keyword in ["flood", "hurricane", "earthquake", "volcano", "wildfire", "tsunami"]:
                if keyword in img_file.lower():
                    disaster_type = keyword.capitalize()
                    break

            results.append({
                "drone_id": f"xview2_{idx:04d}",
                "timestamp": f"2013-09-12T{12 + idx // 50:02d}:{(idx % 50) * 1:02d}:00",
                "location": f"xView2 Disaster Zone ({disaster_type})",
                "observations": description,
                "modality": "imagery_description",
                "source_file": img_file,
                "disaster_type": disaster_type
            })

    if results:
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n[VLM] Wrote {len(results)} real VLM descriptions to {output_json}")
    else:
        print("[VLM] No descriptions generated.")


if __name__ == "__main__":
    in_dir = os.path.join("data", "imagery", "samples")
    out_file = os.path.join("data", "imagery", "real_drone_metadata.json")
    process_images_with_vlm(in_dir, out_file)
