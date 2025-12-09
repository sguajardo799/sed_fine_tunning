# SED with PaSST

This project fine-tunes the PaSST (Patchout faSt Spectrogram Transformer) model for Sound Event Detection (SED) using a custom dataset.

## Installation

1.  **Clone the repository** (if applicable).
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: `torchcodec` is not used due to compatibility issues; `soundfile` is used instead.*

## Data Preparation

The project expects the following data structure:

```
data/
├── global_train_hartf.csv
├── global_val_hartf.csv
└── SigData_1/
    ├── ProjectData_HARTF_front00001.wav
    ├── ProjectData_HARTF_rear00001.wav
    └── ...
```

The CSV files should contain columns: `audio_filename`, `hartf_front_filename`, `hartf_rear_filename`, `class`, `start (s)`, `end (s)`.

## Training

To fine-tune the model:

```bash
python main.py --data_dir data --csv_train data/global_train_hartf.csv --csv_val data/global_val_hartf.csv --epochs 10 --batch_size 8
```

**Arguments:**
- `--data_dir`: Root directory of audio files (default: `data`).
- `--csv_train`: Path to training CSV.
- `--csv_val`: Path to validation CSV.
- `--channel`: Audio channel to use (`front` or `rear`, default: `front`).
- `--epochs`: Number of training epochs.
- `--batch_size`: Batch size.
- `--lr`: Learning rate.
- `--test`: Run a quick test with limited data.

## Evaluation

To evaluate a trained model:

```bash
python evaluate.py --model_path results/best_model.pth --csv_val data/global_val_hartf.csv
```

**Arguments:**
- `--model_path`: Path to the saved model checkpoint.
- `--threshold`: Threshold for binarizing predictions (default: 0.5).
- Other arguments (`--data_dir`, `--channel`, etc.) are similar to `main.py`.

## Results

Training results (checkpoints) are saved in the `results/` directory.
Evaluation metrics (Segment-based and Event-based F1 scores) are printed to the console.
