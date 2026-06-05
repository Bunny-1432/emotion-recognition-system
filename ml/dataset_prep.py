import os
import urllib.request
import zipfile
import pandas as pd
from pathlib import Path
import shutil

DATA_DIR = Path(__file__).parent.parent / "data"
RAVDESS_DIR = DATA_DIR / "ravdess"
ZIP_FILE = DATA_DIR / "ravdess.zip"

ZENODO_URL = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip"

# Emotion mapping for RAVDESS
EMOTION_MAP = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def download_progress(count, block_size, total_size):
    percent = int(count * block_size * 100 / total_size)
    if percent % 10 == 0:
        print(f"\rDownloading dataset... {percent}%", end="")

def download_and_extract():
    if not os.path.exists(RAVDESS_DIR):
        os.makedirs(RAVDESS_DIR)
        
    # Check if we already have files
    wav_files = list(RAVDESS_DIR.glob("**/*.wav"))
    if len(wav_files) >= 1440:
        print("Dataset already downloaded and extracted.")
        return
        
    print(f"Downloading RAVDESS dataset from Zenodo (~1GB)...")
    if not os.path.exists(ZIP_FILE):
        urllib.request.urlretrieve(ZENODO_URL, ZIP_FILE, reporthook=download_progress)
        print("\nDownload complete.")
    else:
        print("Zip file already exists, skipping download.")
        
    print("Extracting ZIP file...")
    with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
        
    print("Flattening directory structure...")
    # The zip creates Actor_01, Actor_02, etc folders. We move all wavs to RAVDESS_DIR
    for actor_dir in DATA_DIR.glob("Actor_*"):
        if actor_dir.is_dir():
            for wav_file in actor_dir.glob("*.wav"):
                dest = RAVDESS_DIR / wav_file.name
                if not dest.exists():
                    shutil.move(str(wav_file), str(dest))
            # Remove empty actor dir
            shutil.rmtree(actor_dir)
            
    print("Extraction complete.")
    
    # Clean up zip file to save space
    if os.path.exists(ZIP_FILE):
        os.remove(ZIP_FILE)

def prepare_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    download_and_extract()

    # Parse filenames to create a dataframe
    data = []
    # Search for all wav files in RAVDESS_DIR
    for filepath in RAVDESS_DIR.glob("*.wav"):
        filename = filepath.name
        parts = filename.split('.')[0].split('-')
        if len(parts) == 7:
            emotion_code = parts[2]
            emotion = EMOTION_MAP.get(emotion_code, 'unknown')
            actor = parts[6]
            data.append({
                'file_path': str(filepath),
                'emotion': emotion,
                'emotion_code': int(emotion_code) - 1, # 0-indexed for PyTorch
                'actor': actor
            })
    
    df = pd.DataFrame(data)
    if len(df) == 0:
        raise ValueError("No valid RAVDESS files found. Did extraction fail?")
        
    print(f"Dataset summary:\n{df['emotion'].value_counts()}")
    
    # Split into train and test
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    train_df.to_csv(DATA_DIR / "train.csv", index=False)
    test_df.to_csv(DATA_DIR / "test.csv", index=False)
    print("Train and test CSVs saved.")
    
    return train_df, test_df

if __name__ == "__main__":
    prepare_dataset()
