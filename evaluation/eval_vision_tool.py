"""
evaluation/eval_vision_tool.py

Building-level benchmark for Tool #7 (analyze_disaster_image, task="classify").

Why this design
---------------
The earlier scene-level benchmark was invalid: xView2 labels are per-building
polygons, but the metric collapsed them to one scene label via "most severe
building present", while the VLM described the whole scene. That measured a
label-definition mismatch, not damage-classification ability.

This version:
  - parses building polygons (pixel coordinates) from the xView2 label JSONs
  - crops each labeled building from the matching post-disaster image
  - asks the vision model for an explicit 4-class label (task="classify")
  - samples buildings in a class-balanced way (capped per class)
  - reports building-level accuracy, macro-F1, and latency

No hardcoded results. Ground truth and predictions both come from real artifacts.
Run:
    python evaluation/eval_vision_tool.py --per-class 8
    python evaluation/eval_vision_tool.py --per-class 2   # smoke test
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import tarfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTHONUTF8", "1")
# The production tool unloads the VLM after each call. Benchmarks issue many
# sequential calls, so keep the model resident; otherwise every call pays a
# multi-gigabyte reload.
os.environ.setdefault("OLLAMA_VISION_KEEP_ALIVE", "30m")

META = ROOT / "data" / "imagery" / "real_drone_metadata.json"
TAR = ROOT / "data" / "imagery" / "test_images_labels_targets.tar"
SAMPLES = ROOT / "data" / "imagery" / "samples"
# The tool resolves image_filename against SAMPLES directly, so crops must
# live there (not in a subdirectory) with an unambiguous prefix.
CROPS = SAMPLES
OUT_JSON = ROOT / "evaluation" / "vision_benchmark_results.json"
OUT_MD = ROOT / "evaluation" / "vision_benchmark_report.md"
CLASSES = ("no-damage", "minor-damage", "major-damage", "destroyed")


def wkt_bbox(wkt: str):
    """Extract pixel-space bounding box from a POLYGON WKT string."""
    nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", wkt)]
    if len(nums) < 8:
        return None
    xs, ys = nums[0::2], nums[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def collect_buildings(tar, metadata):
    """Return list of (crop_path, true_class, source_scene) for labeled buildings."""
    from PIL import Image
    CROPS.mkdir(parents=True, exist_ok=True)
    out = []
    for item in metadata:
        fn = item.get("source_file", "")
        if "_post_disaster" not in fn:
            continue
        member = "test/labels/" + fn.replace(".png", ".json")
        if member not in tar.getnames():
            continue
        img_path = SAMPLES / fn
        if not img_path.exists():
            continue
        data = json.load(tar.extractfile(member))
        feats = data.get("features", {}).get("xy", [])
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        w, h = img.size
        for feat in feats:
            props = feat.get("properties", {})
            label = props.get("subtype")
            if label not in CLASSES:
                continue
            bbox = wkt_bbox(feat.get("wkt", ""))
            if not bbox:
                continue
            x0, y0, x1, y1 = bbox
            # The building polygon alone is usually only ~60px across. A vision
            # model cannot judge damage from that alone; include surrounding
            # context so the crop is at least MIN_CROP px on each side.
            MIN_CROP = 192
            bw, bh = x1 - x0, y1 - y0
            pad = max(MIN_CROP / 2 - bw / 2, MIN_CROP / 2 - bh / 2, bw * 0.5)
            x0 = max(0, int(x0 - pad)); y0 = max(0, int(y0 - pad))
            x1 = min(w, int(x1 + pad)); y1 = min(h, int(y1 + pad))
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            crop_name = f"bench_{item.get('drone_id','x')}_{props.get('uid','')[:8]}.png"
            crop_path = CROPS / crop_name
            img.crop((x0, y0, x1, y1)).save(crop_path)
            out.append((crop_name, label, fn))
    return out


def macro_f1(pairs):
    stats = {}
    for c in CLASSES:
        tp = sum(1 for g, p in pairs if g == c and p == c)
        fp = sum(1 for g, p in pairs if g != c and p == c)
        fn = sum(1 for g, p in pairs if g == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        stats[c] = {"precision": round(prec, 4), "recall": round(rec, 4),
                    "f1": round(f1, 4), "support": tp + fn}
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    metadata = json.loads(META.read_text(encoding="utf-8"))
    tar = tarfile.open(TAR)
    buildings = collect_buildings(tar, metadata)
    print(f"Labeled building crops: {len(buildings)}")

    by_class = defaultdict(list)
    for name, label, scene in buildings:
        by_class[label].append((name, label, scene))
    rng = random.Random(args.seed)
    sample = []
    for c in CLASSES:
        pool = by_class[c][:]
        rng.shuffle(pool)
        sample.extend(pool[: args.per_class])
    rng.shuffle(sample)
    print(f"Sampled {len(sample)} crops (cap {args.per_class}/class)")

    from tools.vision_tool import analyze_disaster_image
    rows = []
    for i, (name, truth, scene) in enumerate(sample, 1):
        t0 = time.perf_counter()
        try:
            raw = analyze_disaster_image.invoke({"image_filename": name, "task": "classify"})
        except Exception as exc:
            raw = f"VISION ERROR: {exc}"
        latency = time.perf_counter() - t0
        pred = "UNCLASSIFIED"
        m = re.search(r"DAMAGE_CLASS\s*:\s*([a-z-]+)", raw, re.IGNORECASE)
        if m and m.group(1).lower() in CLASSES:
            pred = m.group(1).lower()
        rows.append({"crop": name, "scene": scene, "ground_truth": truth,
                     "predicted": pred, "latency_s": round(latency, 3),
                     "raw_output": raw[:400]})
        print(f"[{i}/{len(sample)}] {truth:14s} -> {pred:14s} {latency:.1f}s {name}")

    all_pairs = [(r["ground_truth"], r["predicted"]) for r in rows]
    assessable = [p for p in all_pairs if p[1] != "UNCLASSIFIED"]
    # Primary metrics count UNCLASSIFIED / tool errors as wrong answers.
    # Dropping them from the denominator would inflate accuracy whenever the
    # tool fails, which is exactly the failure mode this suite must detect.
    acc = sum(1 for g, p in all_pairs if g == p) / len(all_pairs) if all_pairs else 0.0
    acc_assessable = (sum(1 for g, p in assessable if g == p) / len(assessable)
                      if assessable else 0.0)
    pc = macro_f1(all_pairs)
    macro = sum(v["f1"] for v in pc.values()) / len(pc) if pc else 0.0
    lat = [r["latency_s"] for r in rows]
    p95 = sorted(lat)[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else 0.0

    summary = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "tool": "analyze_disaster_image(task=classify)",
        "model": os.getenv("OLLAMA_VISION_MODEL", "qwen3-vl:4b"),
        "label_source": "xView2 building polygons (per-building subtypes)",
        "granularity": "building crop",
        "sampling": f"balanced cap {args.per_class} per class, seed {args.seed}",
        "indexed_records": len(metadata),
        "labeled_buildings": len(buildings),
        "sampled": len(sample),
        "assessable": len(assessable),
        "unclassified": len(rows) - len(assessable),
        "accuracy": round(acc, 4), "accuracy_assessable": round(acc_assessable, 4),
        "macro_f1": round(macro, 4),
        "mean_latency_s": round(sum(lat) / len(lat), 3) if lat else 0.0,
        "p95_latency_s": round(p95, 3), "per_class": pc, "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# Vision Tool Benchmark (building-level)", "",
             f"Generated: {summary['generated']}", f"Model: {summary['model']}",
             f"Labeled buildings: {len(buildings)} | Sampled: {len(sample)} | "
             f"Assessable: {summary['assessable']} | Unclassified: {summary['unclassified']}", "",
             "| Metric | Value |", "|---|---|",
             f"| Accuracy (errors counted) | {acc:.4f} |",
             f"| Accuracy (assessable only) | {acc_assessable:.4f} |",
             f"| Macro-F1 (errors counted) | {macro:.4f} |",
             f"| Mean latency (s) | {summary['mean_latency_s']} |",
             f"| p95 latency (s) | {summary['p95_latency_s']} |", "",
             "| Class | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"]
    for c, v in pc.items():
        lines.append(f"| {c} | {v['precision']} | {v['recall']} | {v['f1']} | {v['support']} |")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nAccuracy={acc:.4f} MacroF1={macro:.4f} latency={summary['mean_latency_s']}s")
    print(f"Wrote {OUT_JSON.name} and {OUT_MD.name}")


if __name__ == "__main__":
    main()
