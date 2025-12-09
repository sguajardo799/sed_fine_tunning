import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import soundfile as sf
from pathlib import Path

class PaSSTDataset(Dataset):
    def __init__(self, csv_path=None, data_dir=None, channel='front', sr=32000, duration=10.0, time_resolution=0.0, hf_dataset=None):
        """
        Args:
            csv_path: Path to the CSV file (optional if hf_dataset is provided).
            data_dir: Root directory of the dataset (optional if hf_dataset is provided).
            channel: 'front' or 'rear'.
            sr: Sampling rate.
            duration: Audio duration in seconds.
            time_resolution: Time resolution of the model output in seconds (e.g., 0.02s).
                             If 0, returns weak labels (not implemented for SED).
            hf_dataset: A Hugging Face Dataset object (optional).
        """
        self.channel = channel
        self.sr = sr
        self.duration = duration
        self.time_resolution = time_resolution
        self.hf_dataset = hf_dataset
        
        if self.hf_dataset is not None:
            # HF mode
            # We assume the HF dataset has the same structure as the CSV
            # But 'datasets' library handles grouping differently.
            # Actually, the HF dataset is flat (one row per event usually, unless we grouped it before uploading).
            # Wait, my upload script creates a dataset from the CSV directly.
            # The CSV has one row per event.
            # So the HF dataset will have one row per event.
            # But we want to train on *files* (which may have multiple events).
            # So we need to group the HF dataset by 'audio_filename' as well.
            
            # This is tricky with HF datasets because they are arrow tables.
            # We can convert to pandas to group, but that defeats the purpose of streaming/lazy loading if the dataset is huge.
            # However, for this task, the dataset seems small enough to fit in memory (CSV based).
            # Let's convert to pandas for grouping logic, but keep the audio loading via HF features if possible?
            # No, if we convert to pandas, we lose the Audio feature lazy loading unless we are careful.
            
            # Alternative: The user wants to use "ds from hub".
            # If I upload the CSV as is, it's event-based.
            # If I want file-based access, I should probably group it *before* creating the dataset or handle it here.
            
            # Let's do the grouping here using pandas on the metadata columns.
            # We can access the underlying table or just convert columns to pandas.
            self.df = self.hf_dataset.to_pandas()
            # The Audio columns in pandas will be dicts or paths depending on how it was loaded.
            # If we used cast_column, they might be decoded.
            # But we want to use the HF dataset for audio loading to leverage its capabilities (streaming etc).
            
            # Actually, if we use `hf_dataset[idx]`, it returns the row with audio decoded.
            # But we need to group by filename.
            
            # Let's use the same logic: group by filename.
            self.classes = sorted(self.df['class'].unique())
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
            
            self.file_groups = self.df.groupby('audio_filename')
            self.filenames = list(self.file_groups.groups.keys())
            
        else:
            # Local mode
            self.df = pd.read_csv(csv_path)
            self.data_dir = Path(data_dir)
            
            # Unique classes
            self.classes = sorted(self.df['class'].unique())
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
            
            # Group by filename to handle multiple events per file
            self.file_groups = self.df.groupby('audio_filename')
            self.filenames = list(self.file_groups.groups.keys())

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        group = self.file_groups.get_group(filename)
        
        # Determine file path/audio based on channel
        if self.channel == 'front':
            col_name = 'hartf_front_filename'
        elif self.channel == 'rear':
            col_name = 'hartf_rear_filename'
        else:
            raise ValueError(f"Invalid channel: {self.channel}")
            
        if self.hf_dataset is not None:
            # HF mode
            # We need to get the audio from the HF dataset.
            # The group contains indices of the events in the original dataframe/dataset.
            # We can pick the first event's index to get the audio file (since they share the same audio file).
            first_idx = group.index[0]
            item = self.hf_dataset[int(first_idx)] # Access by integer index
            
            # item[col_name] should be a dict {'array': ..., 'sampling_rate': ...}
            audio_data = item[col_name]
            wav_np = audio_data['array']
            sr = audio_data['sampling_rate']
            
            wav = torch.from_numpy(wav_np).float()
            # HF audio is usually (channels, time) or just (time,).
            # If mono, it's (time,).
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
            
        else:
            # Local mode
            rel_path = group.iloc[0][col_name]
            wav_path = self.data_dir / rel_path
            
            # Load audio using soundfile
            wav_np, sr = sf.read(wav_path)
            wav = torch.from_numpy(wav_np).float()
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
            else:
                wav = wav.t() # (time, channels) -> (channels, time)
        
        # Resample if necessary
        if sr != self.sr:
            resampler = torchaudio.transforms.Resample(sr, self.sr)
            wav = resampler(wav)
            
        # Ensure mono and squeeze channel dim
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0)
        else:
            wav = wav.squeeze(0)
            
        # Pad or truncate to duration
        target_len = int(self.sr * self.duration)
        if wav.shape[0] < target_len:
            wav = torch.nn.functional.pad(wav, (0, target_len - wav.shape[0]))
        elif wav.shape[0] > target_len:
            wav = wav[:target_len]
            
        # Generate labels
        # If time_resolution is set, generate strong labels
        if self.time_resolution > 0:
            num_frames = int(self.duration / self.time_resolution)
            labels = torch.zeros((num_frames, len(self.classes)))
            
            for _, row in group.iterrows():
                class_idx = self.class_to_idx[row['class']]
                start_frame = int(row['start (s)'] / self.time_resolution)
                end_frame = int(row['end (s)'] / self.time_resolution)
                
                # Clip to valid range
                start_frame = max(0, min(start_frame, num_frames - 1))
                end_frame = max(0, min(end_frame, num_frames))
                
                labels[start_frame:end_frame, class_idx] = 1.0
        else:
            # Weak labels (just in case)
            labels = torch.zeros(len(self.classes))
            for _, row in group.iterrows():
                class_idx = self.class_to_idx[row['class']]
                labels[class_idx] = 1.0
                
        return wav, labels, filename

from torch.utils.data import IterableDataset
from huggingface_hub import hf_hub_url
import requests
import io
import itertools

class StreamingPaSSTDataset(IterableDataset):
    def __init__(self, hf_dataset, repo_id, token=None, channel='front', sr=32000, duration=10.0, time_resolution=0.0, max_samples=None):
        self.hf_dataset = hf_dataset
        self.repo_id = repo_id
        self.token = token
        self.channel = channel
        self.sr = sr
        self.duration = duration
        self.time_resolution = time_resolution
        self.max_samples = max_samples
        
        # We need classes to map to indices. 
        # Since we are streaming, we might not know all classes upfront unless provided.
        # For now, we assume standard classes or we need to scan the dataset (expensive).
        # Or we can hardcode them if known, or pass them in.
        # Let's assume they are passed or we can get them from features if available.
        # The user's CSV has 'class' column.
        # Let's try to get unique classes from the dataset features if possible, 
        # otherwise we might need a list.
        # For this specific dataset, let's look at what we found in previous steps or assume a fixed set.
        # In the non-streaming dataset, we did `sorted(self.df['class'].unique())`.
        # We can't do that easily here.
        # Let's hardcode the classes observed in the CSV head for now or add an argument.
        # Observed classes: 'bell', 'cutlery', 'laugh', 'speech-target', 'voice'
        # Wait, there might be more.
        # Let's add a `classes` argument to __init__.
        self.classes = ['bell', 'cutlery', 'dishes', 'door', 'laugh', 'speech-target', 'voice', 'walk'] # Common SED classes + hartf specific?
        # Actually, let's try to fetch features from hf_dataset if it has ClassLabel.
        # If not, we might default to a known list or require it.
        # Let's use a safe default list based on the user's data we saw.
        # We saw: speech-target, voice, bell, laugh, cutlery.
        # Let's stick to a generic list or ask the user? 
        # Better: pass it from main.py where we might know it or just use the ones we find (but that breaks consistency).
        # Let's assume the user provides it or we use a comprehensive list.
        # I'll add a `classes` argument.
        self.classes = sorted(['bell', 'cutlery', 'dishes', 'door', 'laugh', 'speech-target', 'voice', 'walk', 'water', 'cry']) # Example list
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def set_classes(self, classes):
        self.classes = sorted(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __iter__(self):
        # Group by audio_filename
        # We assume the dataset is sorted by audio_filename!
        
        iterator = iter(self.hf_dataset)
        grouped_iterator = itertools.groupby(iterator, key=lambda x: x['audio_filename'])
        
        if self.max_samples is not None:
            grouped_iterator = itertools.islice(grouped_iterator, self.max_samples)
        
        for filename, group in grouped_iterator:
            rows = list(group)
            if not rows:
                continue
                
            # Get audio from first row
            first_row = rows[0]
            
            if self.channel == 'front':
                audio_rel_path = first_row['hartf_front_filename']
            else:
                audio_rel_path = first_row['hartf_rear_filename']
                
            # Construct URL
            # audio_rel_path might be "SigData_1/..."
            url = hf_hub_url(repo_id=self.repo_id, filename=audio_rel_path, repo_type='dataset')
            
            # Download audio
            headers = {}
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"
                
            try:
                response = requests.get(url, headers=headers)
                # Check for 404 or other errors
                if response.status_code == 404:
                    print(f"Warning: File not found (404): {audio_rel_path}. Skipping.")
                    continue
                response.raise_for_status()
                wav_np, sr = sf.read(io.BytesIO(response.content))
            except Exception as e:
                print(f"Error loading {audio_rel_path}: {e}. Skipping.")
                continue
                
            wav = torch.from_numpy(wav_np).float()
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
            else:
                wav = wav.t()
                
            # Resample
            if sr != self.sr:
                resampler = torchaudio.transforms.Resample(sr, self.sr)
                wav = resampler(wav)
                
            # Mono
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0)
            else:
                wav = wav.squeeze(0)
                
            # Pad/Truncate
            target_len = int(self.sr * self.duration)
            if wav.shape[0] < target_len:
                wav = torch.nn.functional.pad(wav, (0, target_len - wav.shape[0]))
            elif wav.shape[0] > target_len:
                wav = wav[:target_len]
                
            # Labels
            if self.time_resolution > 0:
                num_frames = int(self.duration / self.time_resolution)
                labels = torch.zeros((num_frames, len(self.classes)))
                
                for row in rows:
                    if row['class'] in self.class_to_idx:
                        class_idx = self.class_to_idx[row['class']]
                        start_frame = int(row['start (s)'] / self.time_resolution)
                        end_frame = int(row['end (s)'] / self.time_resolution)
                        
                        start_frame = max(0, min(start_frame, num_frames - 1))
                        end_frame = max(0, min(end_frame, num_frames))
                        
                        labels[start_frame:end_frame, class_idx] = 1.0
            else:
                labels = torch.zeros(len(self.classes))
                for row in rows:
                    if row['class'] in self.class_to_idx:
                        class_idx = self.class_to_idx[row['class']]
                        labels[class_idx] = 1.0
                        
            yield wav, labels, filename
