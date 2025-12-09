import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from src.dataset import PaSSTDataset
from src.model import FineTunePaSST
from torch.utils.data import DataLoader
import sed_eval
import dcase_util

def evaluate(model, loader, device, threshold=0.5, time_resolution=0.1):
    model.eval()
    
    # Store all predictions and ground truths
    # We need to reconstruct the events from frame-level predictions
    
    # List of dicts: {'filename': str, 'event_label': str, 'onset': float, 'offset': float}
    pred_events = []
    gt_events = []
    
    # For segment-based metrics, we can also collect frame-level predictions
    # But sed_eval works with event lists or file lists.
    
    # Let's use sed_eval's segment-based evaluation which takes event lists.
    
    with torch.no_grad():
        for wavs, labels, filenames in tqdm(loader, desc="Evaluating"):
            wavs = wavs.to(device)
            
            # Forward
            logits = model(wavs) # [Batch, Time, Classes]
            probs = torch.sigmoid(logits)
            
            # Binarize
            preds = probs > threshold
            
            # Convert to events
            batch_size = wavs.shape[0]
            for b in range(batch_size):
                filename = filenames[b]
                pred_b = preds[b].cpu().numpy() # [Time, Classes]
                
                # Iterate over classes
                for class_idx in range(pred_b.shape[1]):
                    class_name = loader.dataset.classes[class_idx]
                    
                    # Find continuous segments
                    # Simple run-length encoding or similar
                    is_active = pred_b[:, class_idx]
                    
                    # Find changes
                    diff = np.diff(np.concatenate(([0], is_active.astype(int), [0])))
                    onsets = np.where(diff == 1)[0]
                    offsets = np.where(diff == -1)[0]
                    
                    for on, off in zip(onsets, offsets):
                        onset_time = on * time_resolution
                        offset_time = off * time_resolution
                        
                        pred_events.append({
                            'event_label': class_name,
                            'onset': onset_time,
                            'offset': offset_time,
                            'filename': filename
                        })
                        
    # Load ground truth from CSV directly to ensure we have all events
    # The dataset might chop things or we might want the original GT.
    # But let's use the dataset's DF for consistency.
    df = loader.dataset.df
    for _, row in df.iterrows():
        gt_events.append({
            'event_label': row['class'],
            'onset': row['start (s)'],
            'offset': row['end (s)'],
            'filename': row['audio_filename']
        })
        
    # Evaluate
    # Create file list
    all_files = sorted(list(set(e['filename'] for e in gt_events)))
    
    # Segment-based metrics (1s segment)
    segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(
        event_label_list=loader.dataset.classes,
        time_resolution=1.0
    )
    
    # Event-based metrics
    event_based_metrics = sed_eval.sound_event.EventBasedMetrics(
        event_label_list=loader.dataset.classes,
        t_collar=0.200,
        percentage_of_length=0.2
    )
    
    # Add data
    # Evaluate file by file
    for filename in all_files:
        # Filter events for this file
        file_pred_events = [e for e in pred_events if e['filename'] == filename]
        file_gt_events = [e for e in gt_events if e['filename'] == filename]
        
        segment_based_metrics.evaluate(file_pred_events, file_gt_events)
        event_based_metrics.evaluate(file_pred_events, file_gt_events)
    
    print("\nSegment-based Metrics (1.0s):")
    print(segment_based_metrics)
    
    print("\nEvent-based Metrics:")
    print(event_based_metrics)
    
    return segment_based_metrics, event_based_metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate PaSST for SED")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--csv_val", type=str, default="data/global_val_hartf.csv", help="Path to val CSV")
    parser.add_argument("--channel", type=str, default="front", choices=["front", "rear"], help="Channel to use")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binarization")
    parser.add_argument("--test", action="store_true", help="Run a quick test on few samples")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    # We need to know num_classes. Let's peek at the CSV first.
    temp_df = pd.read_csv(args.csv_val)
    classes = sorted(temp_df['class'].unique())
    num_classes = len(classes)
    
    model = FineTunePaSST(num_classes=num_classes)
    
    # Load checkpoint
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    
    time_res = model.get_time_resolution()
    print(f"Time resolution: {time_res}s")
    
    # Dataset
    val_dataset = PaSSTDataset(args.csv_val, args.data_dir, channel=args.channel, time_resolution=time_res)
    
    if args.test:
        print("Running in test mode (first 10 files)...")
        val_dataset.filenames = val_dataset.filenames[:10]
        
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    evaluate(model, val_loader, device, threshold=args.threshold, time_resolution=time_res)

if __name__ == "__main__":
    main()
