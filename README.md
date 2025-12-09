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
- `--device`: Device to use (e.g., `cuda:0`, `cpu`).
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

## Docker

To run the project using Docker:

1.  **Build the image**:
    ```bash
    docker build -t sed-passt .
    ```

2.  **Run training**:
    Mount the current directory to `/workspace` to access data and save results.
    ```bash
    docker run --gpus all -v $(pwd):/workspace sed-passt python main.py --epochs 10
    ```

3.  **Run evaluation**:
    ```bash
    docker run --gpus all -v $(pwd):/workspace sed-passt python evaluate.py --model_path results/best_model.pth
    ```

**Note:** Ensure your data is in the `data/` directory relative to where you run the command.

## Hugging Face Dataset Integration

### Uploading Dataset

To upload your local dataset to the Hugging Face Hub:

```bash
python upload_ds.py --repo_id <username>/<dataset_name> --token <your_hf_token>
```

**Arguments:**
- `--repo_id`: The ID of the repository to push to (e.g., `user/sed-dataset`).
- `--token`: Your Hugging Face authentication token (optional if already logged in).
- `--data_dir`, `--csv_train`, `--csv_val`: Paths to your local data (defaults to `data/` and standard CSV names).

### Training from Hub

To train directly using a dataset from the Hugging Face Hub:

```bash
python main.py --hf_dataset <username>/<dataset_name> --streaming --token <your_hf_token>
```

This will stream the data directly from the Hub, eliminating the need for local data storage during training.

**Arguments:**
- `--hf_dataset`: Hugging Face Dataset ID.
- `--streaming`: Enable streaming mode (required for large datasets to avoid download).
- `--token`: Hugging Face token (required for private datasets or higher rate limits).

