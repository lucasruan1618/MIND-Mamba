# MIND-Mamba

## Project Overview
This project implements a Recommender System (RecSys) for the **MIND (Microsoft News Recommendation)** dataset, utilizing the **Mamba** state space model architecture. It appears to be a research or experimental integration of Mamba into a news recommendation pipeline, evolved from a codebase originally designed for "Fastformer".

The core objective is to leverage Mamba's efficient sequence modeling capabilities for user history encoding and news recommendation.

## Key Components

### 1. Data Preparation
*   **Script:** `data_generation.py`
*   **Purpose:** Converts the raw MIND dataset (train, dev, test) into the format required by the training pipeline (`speedy_data` format).
*   **Output:** Generates `docs.tsv` and `ProtoBuf_*.tsv` files in `./data/speedy_data/`.

### 2. Models
*   **Mamba:**
    *   `models/mamba.py`: Core implementation of the Mamba architecture (MambaBlock, ResidualBlock, SSM) and `MambaMIND` wrapper.
    *   `models/mamba_speedyrec.py`: specialized implementation of the Mamba model adapted for the SpeedyRec-style training loop used in this project.
*   **Fastformer (Legacy/Baseline):**
    *   `models/fast.py`: Implementation of the Fastformer model.

### 3. Training & Evaluation
*   **Main Training Script:** `train_mamba.py` (Note: `train.py` likely trains the Fastformer baseline).
*   **Submission/Inference:** `submission_mamba.py` (Generates predictions for the leaderboard).
*   **Configuration:** `parameters.py` defines all hyperparameters (learning rate, batch size, model dimensions) and file paths.

## Setup & Usage

### Prerequisites
*   Python 3.6+
*   PyTorch
*   Transformers
*   Scikit-learn

### Step 1: Data Preparation
Download the MIND dataset (Large or Small) and extract it. Then run the generation script:

```bash
python data_generation.py --raw_data_path /path/to/raw/MIND/data
```
*   Ensure the raw data path contains `MINDlarge_train`, `MINDlarge_dev`, and `MINDlarge_test` folders.

### Step 2: Training Mamba
To train the Mamba-based model, use `train_mamba.py`. You can customize hyperparameters via command-line arguments (defined in `parameters.py`).

```bash
python train_mamba.py \
    --root_data_dir ./data/speedy_data/ \
    --epochs 6 \
    --batch_size 64 \
    --lr 0.0001 \
    --news_dim 64 \
    --savename mamba_mind_run
```

**Key Arguments:**
*   `--root_data_dir`: Path to the processed data (default: `./data/speedy_data/`).
*   `--news_attributes`: Attributes to use (e.g., `title`, `abstract`).
*   `--pretreained_model`: Option to use `unilm` or other PLMs (though Mamba might be trained from scratch or differently).

### Step 3: Prediction / Submission
To generate a submission file (zip) for the MIND leaderboard:

```bash
python submission_mamba.py \
    --root_data_dir ./data/speedy_data/ \
    --load_ckpt_name ./saved_models/your_best_model.pt
```

## Directory Structure
*   `data_handler/`: Scripts for data loading, preprocessing, and streaming.
    *   `TrainDataloader.py` / `TestDataloader.py`: Custom dataloaders for the project's specific format.
*   `models/`: Model definitions (Mamba, Fastformer, etc.).
*   `utility/`: Helper functions for metrics (`metrics.py`) and general utilities (`utils.py`).
*   `saved_models/`: Default directory where trained checkpoints are saved.

## Notes for Developers
*   **Legacy Code:** `README.md` describes the original Fastformer project. Use it for context but rely on `train_mamba.py` and `models/mamba.py` for the current Mamba implementation.
*   **Distributed Training:** The training script `train_mamba.py` is set up for `DistributedDataParallel` (DDP) by default.
