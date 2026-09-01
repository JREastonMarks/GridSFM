import argparse
import os
import shutil

def filter_ma_data(input_dir="data/raw", output_dir="data/processed/ma_data"):
    print(f"Isolating Massachusetts and New England dataset from {input_dir}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # We want files that relate to Massachusetts and New England
    target_prefixes = ['massachusetts_', 'new_england_']
    
    # Iterate through the time horizons (04h, 16h)
    for time_horizon in ['04h', '16h']:
        horizon_path = os.path.join(input_dir, time_horizon)
        if not os.path.exists(horizon_path):
            continue
            
        out_horizon_path = os.path.join(output_dir, time_horizon)
        os.makedirs(out_horizon_path, exist_ok=True)
        
        for file in os.listdir(horizon_path):
            if any(file.startswith(prefix) for prefix in target_prefixes):
                src = os.path.join(horizon_path, file)
                dst = os.path.join(out_horizon_path, file)
                shutil.copy2(src, dst)
                print(f"Copied {file} to {out_horizon_path}")
                
    print(f"Filtered dataset saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data/raw")
    parser.add_argument("--output_dir", type=str, default="data/processed/ma_data")
    args = parser.parse_args()
    
    filter_ma_data(args.input_dir, args.output_dir)
