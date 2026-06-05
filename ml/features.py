import librosa
import numpy as np
import random

def spec_augment(features, time_masking=True, freq_masking=True):
    """Applies SpecAugment-like masking to features (Time and Frequency)."""
    aug_features = features.copy()
    num_freqs, num_frames = aug_features.shape
    
    if freq_masking:
        f = random.randint(0, min(30, num_freqs))
        f0 = random.randint(0, max(1, num_freqs - f))
        aug_features[f0:f0+f, :] = 0
        
    if time_masking:
        t = random.randint(0, min(30, num_frames))
        t0 = random.randint(0, max(1, num_frames - t))
        aug_features[:, t0:t0+t] = 0
        
    return aug_features

def extract_features(file_path, max_len=180, augment=False):
    """
    Extracts MFCC, Chroma, Mel Spectrogram, ZCR, and RMS from an audio file.
    Args:
        file_path (str): Path to the audio file.
        max_len (int): Maximum length of the feature vector sequence.
        augment (bool): Whether to apply SpecAugment.
    Returns:
        np.array: Extracted feature matrix of shape (features, time_steps)
    """
    try:
        # Load audio file
        audio, sample_rate = librosa.load(file_path, sr=22050)
        
        # Extract features
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        
        stft = np.abs(librosa.stft(audio))
        chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
        
        mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
        
        zcr = librosa.feature.zero_crossing_rate(audio)
        rms = librosa.feature.rms(y=audio)
        
        # Concatenate features along the feature dimension
        # MFCC(40) + Chroma(12) + Mel(128) + ZCR(1) + RMS(1) = 182 features
        features = np.concatenate((mfccs, chroma, mel, zcr, rms), axis=0)
        
        # Pad or truncate to max_len
        if features.shape[1] > max_len:
            features = features[:, :max_len]
        else:
            pad_width = max_len - features.shape[1]
            features = np.pad(features, pad_width=((0, 0), (0, pad_width)), mode='constant')
            
        if augment:
            features = spec_augment(features)
            
        return features
    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}")
        print(e)
        return None
