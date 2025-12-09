import argparse
import torch
from pathlib import Path
from src.dataset import PaSSTDataset
from src.model import FineTunePaSST
from src.train import train
from torch.utils.data import DataLoader
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser(description="Fine-tune PaSST for SED")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--csv_train", type=str, default="data/global_train_hartf.csv", help="Path to train CSV")
    parser.add_argument("--csv_val", type=str, default="data/global_val_hartf.csv", help="Path to val CSV")
    parser.add_argument("--hf_dataset", type=str, help="Hugging Face Dataset ID (e.g. user/dataset)")
    parser.add_argument("--channel", type=str, default="front", choices=["front", "rear"], help="Channel to use")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--test", action="store_true", help="Run a quick test")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model to get time resolution
    # We need to instantiate it to check resolution or use the estimated one
    model = FineTunePaSST(num_classes=7) # Num classes will be updated or checked
    time_res = model.get_time_resolution()
    print(f"Estimated time resolution: {time_res}s")
    
    # Datasets
    # Datasets
    if args.hf_dataset:
        print(f"Loading dataset from Hugging Face: {args.hf_dataset}")
        # Load dataset
        # Assuming the dataset has 'train' and 'validation' splits
        hf_ds = load_dataset(args.hf_dataset)
        train_dataset = PaSSTDataset(hf_dataset=hf_ds['train'], channel=args.channel, time_resolution=time_res)
        val_dataset = PaSSTDataset(hf_dataset=hf_ds['validation'], channel=args.channel, time_resolution=time_res)
    else:
        train_dataset = PaSSTDataset(args.csv_train, args.data_dir, channel=args.channel, time_resolution=time_res)
        val_dataset = PaSSTDataset(args.csv_val, args.data_dir, channel=args.channel, time_resolution=time_res)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Update model num_classes if different
    if len(train_dataset.classes) != model.num_classes:
        print(f"Updating model num_classes to {len(train_dataset.classes)}")
        model = FineTunePaSST(num_classes=len(train_dataset.classes))
        
    model = model.to(device)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Test mode
    if args.test:
        print("Running test mode...")
        args.epochs = 1
        # Limit datasets
        train_dataset.filenames = train_dataset.filenames[:10]
        val_dataset.filenames = val_dataset.filenames[:10]
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
        
    # Train
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    train(model, train_loader, val_loader, args.epochs, args.lr, device, results_dir)

if __name__ == "__main__":
    main()
