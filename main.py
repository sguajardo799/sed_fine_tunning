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
    
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode")
    parser.add_argument("--token", type=str, help="Hugging Face Token")
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g., 'cuda:0', 'cpu')")
    parser.add_argument("--model", type=str, default="passt", choices=["passt", "crnn"], help="Model to use")
    
    # Pretrained option (default True)
    parser.add_argument("--no_pretrained", action="store_false", dest="pretrained", help="Do not use pre-trained weights")
    parser.set_defaults(pretrained=True)
    
    args = parser.parse_args()
    
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model to get time resolution
    # We need to instantiate it to check resolution or use the estimated one
    if args.model == "passt":
        model = FineTunePaSST(num_classes=7, pretrained=args.pretrained) # Num classes will be updated or checked
    elif args.model == "crnn":
        from src.crnn import CRNN
        model = CRNN(num_classes=7, pretrained=args.pretrained)
        
    time_res = model.get_time_resolution()
    print(f"Estimated time resolution: {time_res}s")
    
    # Datasets
    if args.hf_dataset:
        print(f"Loading dataset from Hugging Face: {args.hf_dataset}")
        
        if args.streaming:
            from src.dataset import StreamingPaSSTDataset
            from huggingface_hub import login
            
            if args.token:
                login(token=args.token)
                
            print("Using streaming mode...")
            hf_ds = load_dataset(args.hf_dataset, streaming=True, token=args.token)
            
            train_dataset = StreamingPaSSTDataset(
                hf_dataset=hf_ds['train'], 
                repo_id=args.hf_dataset,
                token=args.token,
                channel=args.channel, 
                time_resolution=time_res
            )
            val_dataset = StreamingPaSSTDataset(
                hf_dataset=hf_ds['validation'], 
                repo_id=args.hf_dataset,
                token=args.token,
                channel=args.channel, 
                time_resolution=time_res
            )
            
            # For streaming, we can't easily get len() or classes upfront unless we scan or hardcode.
            # We'll rely on the default classes in StreamingPaSSTDataset or update if possible.
            # But we need to make sure model matches.
            if len(train_dataset.classes) != model.num_classes:
                print(f"Updating model num_classes to {len(train_dataset.classes)}")
                if args.model == "passt":
                    model = FineTunePaSST(num_classes=len(train_dataset.classes), pretrained=args.pretrained)
                elif args.model == "crnn":
                    from src.crnn import CRNN
                    model = CRNN(num_classes=len(train_dataset.classes), pretrained=args.pretrained)
                
            model = model.to(device)
            
            # DataLoaders for streaming (no shuffle, num_workers=0 usually safer for iterable but we can try)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0) 
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0)
            
        else:
            # Load dataset
            # Assuming the dataset has 'train' and 'validation' splits
            hf_ds = load_dataset(args.hf_dataset)
            train_dataset = PaSSTDataset(hf_dataset=hf_ds['train'], channel=args.channel, time_resolution=time_res)
            val_dataset = PaSSTDataset(hf_dataset=hf_ds['validation'], channel=args.channel, time_resolution=time_res)
            
            print(f"Train samples: {len(train_dataset)}")
            print(f"Val samples: {len(val_dataset)}")
            
            # Update model num_classes if different
            if len(train_dataset.classes) != model.num_classes:
                print(f"Updating model num_classes to {len(train_dataset.classes)}")
                if args.model == "passt":
                    model = FineTunePaSST(num_classes=len(train_dataset.classes), pretrained=args.pretrained)
                elif args.model == "crnn":
                    from src.crnn import CRNN
                    model = CRNN(num_classes=len(train_dataset.classes), pretrained=args.pretrained)
                
            model = model.to(device)
            
            # DataLoaders
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    else:
        train_dataset = PaSSTDataset(args.csv_train, args.data_dir, channel=args.channel, time_resolution=time_res)
        val_dataset = PaSSTDataset(args.csv_val, args.data_dir, channel=args.channel, time_resolution=time_res)
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        
        # Update model num_classes if different
        if len(train_dataset.classes) != model.num_classes:
            print(f"Updating model num_classes to {len(train_dataset.classes)}")
            if args.model == "passt":
                model = FineTunePaSST(num_classes=len(train_dataset.classes), pretrained=args.pretrained)
            elif args.model == "crnn":
                from src.crnn import CRNN
                model = CRNN(num_classes=len(train_dataset.classes), pretrained=args.pretrained)
            
        model = model.to(device)
        
        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Test mode
    if args.test:
        print("Running test mode...")
        args.epochs = 1
        # Limit datasets
        if hasattr(train_dataset, 'filenames'):
            train_dataset.filenames = train_dataset.filenames[:10]
            val_dataset.filenames = val_dataset.filenames[:10]
        else:
            # For streaming, we can't easily slice. 
            # We can rely on the loop breaking early or use take() if we implemented it,
            # but DataLoader will just iterate.
            # Since we set epochs=1 and it's streaming, it might run for the whole dataset?
            # No, we can just let it run for a bit or rely on the fact that test mode usually implies small data.
            # But here we are streaming the full dataset.
            # We should probably wrap the dataset to limit it or just let the user interrupt.
            # Or better, we can use `itertools.islice` in the dataset if we added a limit method.
            # For now, let's just print a warning.
            print("Warning: Cannot limit streaming dataset size in test mode. It will run for one full epoch (or until stopped).")
            pass
            
        # DataLoader shuffle must be False for IterableDataset
        shuffle_train = not args.streaming
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=shuffle_train)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
        
    # Train
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    train(model, train_loader, val_loader, args.epochs, args.lr, device, results_dir)

if __name__ == "__main__":
    main()
