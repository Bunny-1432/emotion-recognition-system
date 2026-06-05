import pandas as pd
import requests
import time
import os
from tqdm import tqdm

def main():
    # Load test data
    test_csv = "data/test.csv"
    if not os.path.exists(test_csv):
        print(f"Error: {test_csv} not found.")
        return

    df = pd.read_csv(test_csv)
    print(f"Found {len(df)} test samples.")

    # API endpoint
    url = "http://localhost:8000/predict"
    
    # Wait for the API to be ready
    health_url = "http://localhost:8000/health"
    print("Waiting for API to start...")
    api_ready = False
    for _ in range(30):
        try:
            response = requests.get(health_url)
            if response.status_code == 200:
                api_ready = True
                break
        except requests.ConnectionError:
            pass
        time.sleep(1)
        
    if not api_ready:
        print("API did not start in time.")
        return
        
    print("API is ready. Running predictions...")
    
    df = df.head(20)
    correct = 0
    total = len(df)
    
    for _, row in tqdm(df.iterrows(), total=total):
        file_path = row['file_path']
        true_emotion = row['emotion']
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
            
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f, "audio/wav")}
            try:
                response = requests.post(url, files=files)
                if response.status_code == 200:
                    pred = response.json()["prediction"]
                    if pred == true_emotion:
                        correct += 1
                else:
                    print(f"Error for {file_path}: {response.text}")
            except Exception as e:
                print(f"Failed request for {file_path}: {e}")
                
    accuracy = correct / total * 100
    print(f"\nAccuracy on test set: {accuracy:.2f}% ({correct}/{total})")

if __name__ == "__main__":
    main()
