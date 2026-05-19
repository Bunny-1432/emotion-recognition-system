# Speech Emotion Recognition System

An end-to-end deep learning system designed to classify human emotions from speech audio. It features a PyTorch-based ML pipeline, a FastAPI backend, and a modern, 3D-enhanced Next.js frontend UI.

## Features

- **Robust ML Architecture:** Uses a hybrid 1D CNN + LSTM architecture to capture both frequency-domain (spatial) and time-domain (temporal) dependencies from audio features (MFCC, Chroma, Mel Spectrogram).
- **FastAPI Inference Service:** A highly performant REST API that processes audio files and runs predictions efficiently.
- **Next.js & React Three Fiber UI:** A world-class dashboard featuring a morphing 3D sphere that reacts dynamically to the predicted emotion, utilizing an aesthetic color palette and fluid animations.

## Getting Started

### Prerequisites
- Python 3.9+
- Node.js 18+

### Backend Setup (ML Pipeline & API)

1. Navigate to the project root (`emotion_recognition/`).
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # Mac/Linux:
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install torch torchvision torchaudio librosa fastapi uvicorn python-multipart pydantic scikit-learn requests numpy pandas scipy soundfile tqdm matplotlib seaborn
   ```
4. **Train the Model:**
   ```bash
   python ml/train.py
   ```
   *(Note: If the RAVDESS dataset is not found, the script automatically generates synthetic audio data to demonstrate a working pipeline without waiting for a multi-GB download).*

5. **Start the Inference Server:**
   ```bash
   python backend/app.py
   ```
   The API will be available at `http://localhost:8000`.

### Frontend Setup (UI Dashboard)

1. Open a new terminal and navigate to the `frontend/` directory.
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the Next.js development server:
   ```bash
   npm run dev
   ```
4. Open your browser and navigate to `http://localhost:3000`.

## Model Evaluation

You can evaluate the trained model on the held-out test set by running:
```bash
python ml/evaluate.py
```
This script will output a classification report and save a `confusion_matrix.png` in the project root.

## Adding Custom Datasets

To train on your own data or the full RAVDESS dataset:
1. Place your audio files (`.wav`) in `data/ravdess/` (or update paths in `ml/dataset_prep.py`).
2. Ensure `ml/dataset_prep.py` correctly parses your filenames into `emotion_code` and `actor`.
3. Run `python ml/train.py`.
