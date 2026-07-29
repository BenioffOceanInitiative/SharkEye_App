import argparse
import shutil
import sys
from datetime import datetime, timezone
from google.cloud import storage
from sharkeye_app import mass_prediction
from pathlib import Path
from tqdm import tqdm
import os
import json
import pandas as pd

def download_test_videos(
        source_folder: str = "sharkeye_videos/2023_Missed_Sharks_Trimmed",
        destination: Path = Path("logging_results/test_videos"),
        num_videos: int = 6):
    """
    Downloads trimmed videos from cloud bucket.

    `source_folder` is `bucket/prefix` (e.g. sharkeye_videos/2023_Missed_Sharks_Trimmed).
    `list_blobs` takes the bucket name and an optional prefix separately.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    bucket_name, _, prefix = source_folder.partition("/")
    if prefix and not prefix.endswith("/"):
        prefix = prefix + "/"

    try:
        client = storage.Client()
        # Skip the folder placeholder object; only pull video files.
        blobs = [
            b for b in client.list_blobs(bucket_name, prefix=prefix or None)
            if b.name.lower().endswith((".mp4", ".mov"))
        ]
        if num_videos > 0:
            blobs = blobs[:num_videos]

        videos = tqdm(blobs, desc="Downloading test videos")
        video_paths = []
        for b in videos:
            videos.set_description(f"Downloading {Path(b.name).name}")
            file_path = destination / Path(b.name).name
            b.download_to_filename(file_path)
            video_paths.append(file_path)
        print(f"Successfully downloaded {len(video_paths)} video(s) from gs://{source_folder}")
        return video_paths
    except Exception as e:
        print(f"Direct GCS access failed: {e}")
        return

def parse_args(): 
    parser = argparse.ArgumentParser(description="Run headless object tracking on videos.")
    parser.add_argument('--build_name', type=str, help='Name of build to be exported', required=True)
    parser.add_argument('--commit_hash', type=str, help='Github commit hash for compiled build', required=True)
    parser.add_argument('--input_dir', type=str, help='Path to folder of videos for processing. If not provided will default to Google Cloud images')
    parser.add_argument('--num_videos', type=int, default=0, help='Number of videos to process. Limits the number of videos downloaded from Google Cloud if input_dir not provided')
    parser.add_argument('--save_results', action='store_true')
    return parser.parse_args()

def print_logs(all_logs):
    total_tracks = 0
    total_videos = len(all_logs)
    total_footage_length = 0
    total_processing_duration = 0
    total_segmentation_duration = 0

    for video, logs in all_logs.items():
        print(f" Video Name: {Path(video).name}, Video Length(s): {round(logs.get('video_length'), 2)}, # Tracks: {logs.get('total_tracks')}, Procesing Duration: {round(logs.get('total_processing_duration'), 2)}, Time to Segment: {round(logs.get('total_segmentation_duration'), 2)}")
        
        total_footage_length += logs.get('video_length')
        total_tracks += logs.get('total_tracks')
        total_processing_duration += logs.get('total_processing_duration')
        total_segmentation_duration += logs.get('total_segmentation_duration')

    print(f"\nProcessed {total_videos} videos\n Total Footage Length: {round(total_footage_length, 2)}, # Tracks: {total_tracks}, Total Procesing Duration: {round(total_processing_duration, 2)}, Total Time to Segment: {round(total_segmentation_duration, 2)}")

def cleanup_results_folder(results_dir: Path):
    shutil.rmtree(results_dir)

def export_to_dataframe(all_logs: dict, build_name: str, commit_hash: str):
    total_tracks = 0
    total_videos = len(all_logs)
    total_footage_length = 0
    total_processing_duration = 0
    total_segmentation_duration = 0

    for video, logs in all_logs.items():
        total_footage_length += logs.get('video_length')
        total_tracks += logs.get('total_tracks')
        total_processing_duration += logs.get('total_processing_duration')
        total_segmentation_duration += logs.get('total_segmentation_duration')

    results_dict = {
        'build': [build_name],
        'commit-hash': [commit_hash],
        'total_videos': [total_videos],
        'total_footage_length': [total_footage_length],
        'total_tracks': [total_tracks],
        'total_processing_duration': [total_processing_duration],
        'total_segmentation_duration': [total_segmentation_duration],
    }

    return pd.DataFrame(results_dict)

def upload_results_folder(results_dir: Path, build_name: str, bucket):
    """Upload every file under results_dir to logging_results/{build_name}_{utctime}/."""
    utc_stamp = datetime.now(timezone.utc).strftime("%H%M%S")
    folder_name = f"{build_name}_{utc_stamp}"
    prefix = f"logging_results/{folder_name}"

    files = [p for p in results_dir.rglob("*") if p.is_file()]
    uploads = tqdm(files, desc=f"Uploading to gs://sharkeye-app-build/{prefix}")
    for path in uploads:
        relative = path.relative_to(results_dir).as_posix()
        blob = bucket.blob(f"{prefix}/{relative}")
        uploads.set_description(f"Uploading {relative}")
        blob.upload_from_filename(path)

    print(f"Results uploaded to gs://sharkeye-app-build/{prefix}/")
    return prefix


def upload_results(results_dir: Path, build_name: str, commit_hash: str, save_results: bool = False):
    client = storage.Client()
    log_path = results_dir / "processing_logs.json"
    with open(log_path) as f:
        new_logs = json.load(f)

    new_logs_df = export_to_dataframe(all_logs=new_logs, build_name=build_name, commit_hash=commit_hash)

    bucket = client.get_bucket('sharkeye-app-build')
    blob = bucket.blob('build_logs.csv')
    try:
        blob.download_to_filename(results_dir / 'previous_logs.csv')
        previous_logs_df = pd.read_csv(results_dir / 'previous_logs.csv', index_col=0)
        final_logs = pd.concat([new_logs_df, previous_logs_df[::-1]], ignore_index=True)
    except Exception:
        print("Couldn't retrieve previous logs")
        final_logs = new_logs_df

    final_logs.to_csv(results_dir / 'build_logs.csv')
    print("Final logs converted to CSV")
    blob.upload_from_filename(results_dir / 'build_logs.csv')
    print("Uploaded build logs to Cloud Bucket")
    if save_results:
        upload_results_folder(results_dir, build_name, bucket)

def main():
    args = parse_args()
    base_dir = Path("logging_results")
    num_videos = args.num_videos
    save_results = args.save_results
    build_name = args.build_name
    commit_hash = args.commit_hash

    results_dir = base_dir / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.input_dir:
        input_dir = Path(args.input_dir)
        video_paths = [Path(os.path.join(args.input_dir, f)) for f in os.listdir(input_dir)]
        if num_videos > 0:
            video_paths = video_paths[:num_videos]
    else:
        video_paths = download_test_videos(num_videos=num_videos)

    if not video_paths:
        print("No videos to process; aborting.")
        sys.exit(1)

    mass_prediction(video_paths=video_paths, current_output_dir=results_dir)
    with open(results_dir / "processing_logs.json") as f:
        logs = json.load(f)

    print_logs(logs)
    try:
        print("\nUploading Results to Cloud Bucket")
        upload_results(results_dir=results_dir, build_name=build_name, commit_hash=commit_hash, save_results=save_results)
    except Exception as e:
        print(f"Error uploading results: {e}")

    cleanup_results_folder(results_dir)

if __name__ == "__main__":
    main()