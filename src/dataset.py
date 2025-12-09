import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import soundfile as sf
from pathlib import Path

class PaSSTDataset(Dataset):
    def __init__(self, csv_path, data_dir, channel='front', sr=32000, duration=10.0, time_resolution=0.0):
        """
        Args:
            csv_path: Path to the CSV file.
            data_dir: Root directory of the dataset.
            channel: 'front' or 'rear'.
            sr: Sampling rate.
            duration: Audio duration in seconds.
            time_resolution: Time resolution of the model output in seconds (e.g., 0.02s).
                             If 0, returns weak labels (not implemented for SED).
        """
        self.df = pd.read_csv(csv_path)
        self.data_dir = Path(data_dir)
        self.channel = channel
        self.sr = sr
        self.duration = duration
        self.time_resolution = time_resolution
        
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
        
        # Determine file path based on channel
        # The CSV has columns: hartf_front_filename, hartf_rear_filename
        # But the filename in 'audio_filename' column seems to be the base HRTF file.
        # Let's check the CSV structure again.
        # The user said: "global_train_hartf cssv have a labels and asociated input wav files, i need use hartf_front or rear file"
        # The CSV sample showed:
        # audio_filename: SigData_1/ProjectData_HRTF00001.wav
        # hartf_front_filename: SigData_1/ProjectData_HARTF_front00001.wav
        # hartf_rear_filename: SigData_1/ProjectData_HARTF_rear00001.wav
        
        if self.channel == 'front':
            rel_path = group.iloc[0]['hartf_front_filename']
        elif self.channel == 'rear':
            rel_path = group.iloc[0]['hartf_rear_filename']
        else:
            raise ValueError(f"Invalid channel: {self.channel}")
            
        wav_path = self.data_dir / rel_path
        
        # Load audio
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
