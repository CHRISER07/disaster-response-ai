"""
ingestion/extract_xview2_subset.py

Efficiently extracts a subset of real disaster images from the xView2 tar archive
without unpacking the entire 2.7GB file.

Fix log:
  - FIXED: tar.extract(member, path=...) without filter= is deprecated in Python 3.12
    and will become an error in Python 3.14. Added filter='data' for safe extraction.
"""
import tarfile
import os


def extract_subset(tar_path: str, output_dir: str, num_images: int = 50) -> int:
    """
    Streams a small subset of disaster images directly from the tar archive.
    Only extracts real post-disaster images (>10KB, from the images/ subfolder).

    Args:
        tar_path: Path to the xView2 tar archive.
        output_dir: Directory where extracted images will be saved.
        num_images: Maximum number of images to extract.

    Returns:
        Number of images successfully extracted.
    """
    if not os.path.exists(tar_path):
        print(f"[xView2] Archive not found: {tar_path}")
        print("  Download the xView2 dataset from: https://xview2.org/")
        return 0

    os.makedirs(output_dir, exist_ok=True)
    extracted_count = 0

    print(f"[xView2] Opening archive: {os.path.basename(tar_path)}")
    print(f"[xView2] Extracting up to {num_images} disaster images...")

    try:
        with tarfile.open(tar_path, "r") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                if not (member.name.endswith(".png") or member.name.endswith(".jpg")):
                    continue
                # Only grab raw images (not labels/targets)
                if "images/" not in member.name:
                    continue
                # Filter out tiny/dummy images
                if member.size < 10_000:
                    continue

                # Flatten the path — extract to output_dir directly
                member.name = os.path.basename(member.name)

                # FIX: filter='data' is required in Python 3.12+ to avoid DeprecationWarning.
                # 'data' filter strips dangerous metadata (symlinks, absolute paths, etc.)
                tar.extract(member, path=output_dir, filter="data")
                extracted_count += 1

                if extracted_count % 10 == 0:
                    print(f"  Extracted {extracted_count}/{num_images} images...")

                if extracted_count >= num_images:
                    break

        print(f"\n[xView2] Successfully extracted {extracted_count} images to: {output_dir}")
        return extracted_count

    except tarfile.TarError as e:
        print(f"[xView2] Archive error: {e}")
        return extracted_count
    except Exception as e:
        print(f"[xView2] Unexpected error: {e}")
        return extracted_count


if __name__ == "__main__":
    archive_path = os.path.join("data", "imagery", "test_images_labels_targets.tar")
    out_path     = os.path.join("data", "imagery", "samples")
    count = extract_subset(archive_path, out_path, num_images=50)
    print(f"\nDone. {count} images ready for VLM processing.")
    if count > 0:
        print("Next step: python ingestion/vlm_processor.py")
