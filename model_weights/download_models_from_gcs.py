from google.cloud import storage
from pathlib import Path
from tqdm import tqdm
import requests
import os

def download_models_from_gcs():
    """
    Download models from GCS bucket. If direct access fails, fall back to API.
    """
    
    target_folder = "model_weights"
    Path(target_folder).mkdir(parents=True, exist_ok=True)
    
    # Try direct GCS access first
    try:
        client = storage.Client()
        model_blobs = client.list_blobs('sharkeye-app-models')
        models = tqdm(list(model_blobs), desc="Downloading models")
        for b in models:
            models.set_description(f"Downloading {b.name}")
            file_path = f"{target_folder}/{b.name}"
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            b.download_to_filename(file_path)
        print("Successfully downloaded models using direct GCS access")
        return
    except Exception as e:
        print(f"Direct GCS access failed: {e}")
        print("Falling back to API...")
       
    # Call API to get signed URLs
    try:
        # Call API to get signed URLs
        api_url = "https://us-central1-sharkeye-329715.cloudfunctions.net/sign-up?request=models"
        response = requests.get(api_url, params={'request': 'models'})
        response.raise_for_status()
        signed_urls = response.json()
        
        if not signed_urls:
            raise Exception("No models found in API response")
        
        # Download from signed URLs
        models = tqdm(signed_urls.items(), desc="Downloading models")
        for blob_name, signed_url in models:
            models.set_description(f"Downloading {blob_name}")
            file_path = f"{target_folder}/{blob_name}"
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Download from signed URL
            file_response = requests.get(signed_url, stream=True)
            file_response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                for chunk in file_response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        print("Successfully downloaded models using API signed URLs")
    except Exception as e:
        raise Exception(f"Failed to download models from API: {e}")

if __name__ == "__main__":
    download_models_from_gcs() 