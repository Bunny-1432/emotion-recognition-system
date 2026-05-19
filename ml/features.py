import librosa
import numpy as np

def extract_features(file_path, max_len=100):
    """
    Extracts MFCC, Chroma, and Mel Spectrogram features from an audio file.
    Args:
        file_path (str): Path to the audio file.
        max_len (int): Maximum length of the feature vector sequence.
    Returns:
        np.array: Extracted feature matrix of shape (features, time_steps)
    """
    try:
        # Load audio file
        audio, sample_rate = librosa.load(file_path, sr=22050, res_type='kaiser_fast')
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        
        # Extract Chroma
        stft = np.abs(librosa.stft(audio))
        chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
        
        # Extract Mel Spectrogram
        mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
        
        # Concatenate features along the feature dimension
        features = np.concatenate((mfccs, chroma, mel), axis=0) # Shape: (feature_dim, time_steps)
        
        # Pad or truncate to max_len
        if features.shape[1] > max_len:
            features = features[:, :max_len]
        else:
            pad_width = max_len - features.shape[1]
            features = np.pad(features, pad_width=((0, 0), (0, pad_width)), mode='constant')
            
        return features
    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}")
        print(e)
        return None
