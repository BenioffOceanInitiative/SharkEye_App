from google.cloud import storage
from pathlib import Path
import os
import io
import tempfile
import zipfile
import argparse

def main(build_path: Path):
    builds = {
        'SharkEye_macOS_Intel': 'macos_intel',
        'SharkEye_macOS_Silicon': 'macos_silicon',
        'SharkEye_Windows' : 'windows'
        }

    if not build_path.stem.startswith(tuple(builds.keys())):
        print('File must be named properly')
        return 
    
    if not build_path.name.endswith('.zip'):
        print('File must be a .zip')
        return 
    
    client = storage.Client()
    bucket = client.bucket('sharkeye-app-build')
    
    # Find existing build
    build_prefix  = '_'.join(build_path.stem.split('_')[:-1])
    target_folder = builds[build_prefix] 
    
    build_blobs = client.list_blobs('sharkeye-app-build')
    current_build = None
    for b in build_blobs:
        if b.name.startswith(build_prefix):
            current_build = b
    
    # Move build to archive
    if current_build:
        new_archive_path = f"archive/{target_folder}/{current_build.name}"
        bucket.copy_blob(
            current_build, bucket, new_archive_path
        )
        bucket.delete_blob(current_build.name)

    # Upload zip folder to bucket   
    blob_path = build_path.name
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(build_path)
    print(f"{target_folder} Build Uploaded Successfully. Previous {target_folder} build moved to {new_archive_path}")

def parse_args(): 
    parser = argparse.ArgumentParser(description="Upload ")
    parser.add_argument('--build_path', type=str, required=False, help='Filepath to build ZIP file')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if not args.build_path:
        zip_path = "SharkEye_Windows_Test.zip"
        text_filename = "blank.txt"

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(text_filename, "")
        args.build_path = zip_path

    args.build_path = Path(args.build_path)
    main(args.build_path)