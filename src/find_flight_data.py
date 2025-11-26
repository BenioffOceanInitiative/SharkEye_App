# Find every video in base_directory (year)
# Choose parent directory and look for csv files 
# If there's only one video in the directory 

# For 2023
# Download Comparison sheet, save video_names to list 
# Scan only parent directories of these videos for csvs

import os 
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Read Data

def find_flight_data(video_paths: list):
    ''' Takes in list of videos and returns dictionary consisting of csv flight logs paired with the video path as a key'''
    flight_data_dict = dict()
    for video in video_paths:
        flight_data = video.parent.glob("*.csv")
        if len(flight_data) > 0:
            flight_data_dict['video'] = flight_data
    
    return flight_data_dict

def extract_altitude(flight_data_path: Path):
    flight_data = pd.read_csv(flight_data_path)
    altitudes_ft = flight_data['altitude(feet)']
    mean_alt_ft = sum(altitudes_ft) /  len(altitudes_ft)
    return mean_alt_ft * 0.3048

if __name__ == '__main__':
    extract_altitude('2023_2023-12-02_Transect_Dec-2nd-2023-03-54PM-Flight-Airdata.csv')
    # df_2023 = pd.read_csv('./SharkEye_App/segmentation_comparison/2023 Comparison.csv')
    # file_names = [name for name in df_2023['file_name']]
    # base_path = Path("./sharkeye/sharkeye_videos/2023/")

    # video_paths = []
    # for name in set(file_names):
    #     video_paths.extend(base_path.rglob(name))

    # pd.DataFrame(find_flight_data(video_paths)).to_csv('./videos_with_flight_data.csv')
