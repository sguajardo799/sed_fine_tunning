import argparse
from datasets import load_dataset
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--token", type=str, help="HF Token")
    args = parser.parse_args()
    
    print(f"Testing streaming from {args.repo_id}...")
    
    try:
        # Try loading as a generic dataset
        ds = load_dataset(args.repo_id, split="train", streaming=True, token=args.token)
        print("Dataset loaded successfully.")
        
        from datasets import Audio, Features, Value
        
        # Cast to Audio
        # We need to prepend the base URL or ensure datasets knows it's relative to the repo
        # Actually, if we just cast, it might expect local files.
        # Let's see if we can use cast_column.
        
        # Print raw value first
        print("Raw values from first example:")
        first_item = next(iter(ds))
        print(f"  Front Raw: {first_item['hartf_front_filename']}")
        
        # Cast with decode=False
        ds = ds.cast_column("hartf_front_filename", Audio(sampling_rate=32000, decode=False))
        
        print("Iterating over first 2 examples with Audio cast (decode=False)...")
        for i, item in enumerate(ds):
            if i >= 2: break
            print(f"Item {i} keys: {item.keys()}")
            print(f"  Front Audio: {item['hartf_front_filename']}")


                
    except Exception as e:
        print(f"Error loading dataset: {e}")

if __name__ == "__main__":
    main()
