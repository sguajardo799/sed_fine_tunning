import argparse
import pandas as pd
from datasets import Dataset, DatasetDict, Audio, Features, Value
from pathlib import Path
from huggingface_hub import login

def create_dataset(csv_path, data_dir):
    df = pd.read_csv(csv_path)
    data_dir = Path(data_dir)
    
    # We need to construct absolute paths for the audio files so 'datasets' can find them
    # The CSV has 'hartf_front_filename' and 'hartf_rear_filename'
    # We will create two separate audio columns or just one depending on how we want to structure it.
    # Let's keep both columns as Audio features.
    
    # Update paths to be absolute
    df['hartf_front_filename'] = df['hartf_front_filename'].apply(lambda x: str(data_dir / x))
    df['hartf_rear_filename'] = df['hartf_rear_filename'].apply(lambda x: str(data_dir / x))
    
    # Create dataset
    # Define features
    features = Features({
        'audio_filename': Value('string'),
        'hartf_front_filename': Audio(sampling_rate=32000),
        'hartf_rear_filename': Audio(sampling_rate=32000),
        'class': Value('string'),
        'start (s)': Value('float'),
        'end (s)': Value('float')
    })
    
    dataset = Dataset.from_pandas(df, features=features)
    return dataset

def main():
    parser = argparse.ArgumentParser(description="Upload dataset to Hugging Face Hub")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--csv_train", type=str, default="data/global_train_hartf.csv", help="Path to train CSV")
    parser.add_argument("--csv_val", type=str, default="data/global_val_hartf.csv", help="Path to val CSV")
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face Repo ID (e.g. user/dataset)")
    parser.add_argument("--token", type=str, help="Hugging Face Token")
    
    args = parser.parse_args()
    
    if args.token:
        login(token=args.token)
        
    print("Creating train dataset...")
    train_ds = create_dataset(args.csv_train, args.data_dir)
    
    print("Creating validation dataset...")
    val_ds = create_dataset(args.csv_val, args.data_dir)
    
    ds_dict = DatasetDict({
        'train': train_ds,
        'validation': val_ds
    })
    
    print(f"Pushing to {args.repo_id}...")
    ds_dict.push_to_hub(args.repo_id)
    print("Done!")

if __name__ == "__main__":
    main()
