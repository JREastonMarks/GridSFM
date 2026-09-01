import os
from huggingface_hub import snapshot_download
import argparse

def fetch_data(output_dir="data/raw"):
    print(f"Fetching GridSFM dataset from HuggingFace to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Use snapshot_download because the JSON schemas are heterogeneous
    snapshot_download(repo_id="microsoft/GridSFM_US_power_grid", 
                      repo_type="dataset", 
                      local_dir=output_dir,
                      max_workers=8)
    
    print("Dataset downloaded and saved successfully to raw files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch GridSFM dataset")
    parser.add_argument("--output_dir", type=str, default="data/raw", help="Directory to save raw dataset")
    args = parser.parse_args()
    
    fetch_data(args.output_dir)
