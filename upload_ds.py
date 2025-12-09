import argparse
from huggingface_hub import login, upload_large_folder

def main():
    parser = argparse.ArgumentParser(description="Upload dataset folder to Hugging Face Hub")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data directory to upload")
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face Repo ID (e.g. user/dataset)")
    parser.add_argument("--token", type=str, help="Hugging Face Token")
    parser.add_argument("--num_proc", type=int, default=4, help="Number of processes for upload")
    
    args = parser.parse_args()
    
    if args.token:
        login(token=args.token)
        
    print(f"Uploading {args.data_dir} to {args.repo_id}...")
    
    upload_large_folder(
        folder_path=args.data_dir,
        repo_id=args.repo_id,
        repo_type="dataset",
        num_workers=args.num_proc
    )
    
    print("Done!")

if __name__ == "__main__":
    main()
