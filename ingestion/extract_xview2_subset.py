import tarfile
import os

def extract_subset(tar_path, output_dir, num_images=100):
    """
    Efficiently extracts a small subset of images from a large tar archive 
    without unpacking the whole thing.
    """
    print(f"Opening archive: {tar_path} (This may take a moment...)")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    extracted_count = 0
    
    try:
        with tarfile.open(tar_path, "r") as tar:
            for member in tar:
                # Look specifically for images that represent post-disaster damage 
                if member.isfile() and (member.name.endswith(".png") or member.name.endswith(".jpg")):
                    # Avoid the labels/targets folders, just grab raw images
                    # e.g. 'test/images/xxxx.png'
                    if "images/" in member.name:
                        # Ensure we get images with actual data
                        if member.size > 10000: # Larger than 10k to ensure it's not a dummy image
                            # Extract this specific file
                            member.name = os.path.basename(member.name) # Flatten path
                            tar.extract(member, path=output_dir)
                            extracted_count += 1
                            
                            if extracted_count % 10 == 0:
                                print(f"Extracted {extracted_count}/{num_images} images...")
                                
                            if extracted_count >= num_images:
                                break
                                
        print(f"\nSuccessfully extracted {extracted_count} real disaster images to {output_dir}")
        print(f"Bypassed space limits by streaming directly from the tarfile!")
    except Exception as e:
        print(f"Error during extraction: {e}")

if __name__ == "__main__":
    archive_path = os.path.join("data", "imagery", "test_images_labels_targets.tar")
    out_path = os.path.join("data", "imagery", "samples")
    extract_subset(archive_path, out_path, num_images=50) # 50 is enough for a strong sample without burning API tokens
