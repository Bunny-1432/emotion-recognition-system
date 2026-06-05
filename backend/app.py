import sys
import os
import uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np

# Add the ml directory to the path to import features and model
backend_dir = Path(__file__).parent
project_dir = backend_dir.parent
ml_dir = project_dir / "ml"
sys.path.append(str(ml_dir))

from features import extract_features
from model import EmotionRecognitionModel
from dataset_prep import EMOTION_MAP

app = FastAPI(title="Emotion Recognition API", description="Predict emotion from audio files.")

# Allow CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionRecognitionModel(input_dim=182, num_classes=8).to(device)
model_path = project_dir / "best_model.pt"

try:
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Warning: Model could not be loaded. Please train it first. Error: {e}")

# Emotion names ordered by their integer label (0 to 7)
EMOTION_LABELS = list(EMOTION_MAP.values())

@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    if not file.filename.endswith(('.wav', '.mp3', '.ogg')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a WAV, MP3, or OGG file.")
        
    temp_file = backend_dir / f"temp_audio_{uuid.uuid4().hex}.wav"
    
    try:
        # Save uploaded file temporarily
        with open(temp_file, "wb") as f:
            f.write(await file.read())
            
        # Extract features
        features = extract_features(str(temp_file), max_len=180)
        
        if features is None:
            raise HTTPException(status_code=500, detail="Failed to extract features from audio.")
            
        # Run inference
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device) # Add batch dimension
        
        with torch.no_grad():
            outputs = model(features_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
        # Get top prediction
        top_prob, top_idx = torch.max(probabilities, 0)
        
        # Prepare response
        results = []
        for i, prob in enumerate(probabilities):
            results.append({
                "emotion": EMOTION_LABELS[i],
                "confidence": float(prob)
            })
            
        # Sort results by confidence
        results.sort(key=lambda x: x["confidence"], reverse=True)
            
        return {
            "prediction": EMOTION_LABELS[top_idx],
            "confidence": float(top_prob),
            "all_scores": results
        }
        
    finally:
        # Cleanup temporary file
        if temp_file.exists():
            try:
                os.remove(temp_file)
            except PermissionError:
                pass

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model_path.exists()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
