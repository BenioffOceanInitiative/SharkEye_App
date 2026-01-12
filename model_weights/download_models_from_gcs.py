from google.cloud import storage
from pathlib import Path
from tqdm import tqdm

def download_models_from_gcs():
    client = storage.Client()
    model_blobs = client.list_blobs('sharkeye-app-models')
    target_folder = "model_weights"
    models =  tqdm(model_blobs)
    for b in models:
        models.set_description(f"Downloading {b.name}")
        b.download_to_filename(f"{target_folder}/{b.name}")

if __name__ == "__main__":
    download_models_from_gcs() 