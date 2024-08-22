import os
import sys

def get_base_dir():
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))

def get_results_dir():
    if getattr(sys, 'frozen', False):
        # We are running in a bundle
        base_dir = os.path.dirname(sys.executable)
    else:
        # We are running in a normal Python environment
        base_dir = os.path.dirname(os.path.abspath(__file__))
    
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir
  
def resource_path(relative_path):
    """ Get the absolute path to a resource, works for dev and PyInstaller. """
    if getattr(sys, 'frozen', False):
        # Running in a PyInstaller bundle
        base_path = sys._MEIPASS
        # Check if we're in a Mac app bundle
        if sys.platform == 'darwin' and '.app' in base_path:
            # Navigate to the Resources folder in the Mac app bundle
            base_path = os.path.abspath(os.path.join(base_path, '..', 'Resources'))
    else:
        # Running in a normal Python environment
        base_path = os.path.dirname(os.path.abspath(__file__))
        # Move back one directory
        base_path = os.path.abspath(os.path.join(base_path, '..'))
    
    return os.path.join(base_path, relative_path)

def get_video_path(video_name):
    """
    Get the correct path for a video file.
    First, check if the video exists in the originally selected location.
    If not found, check in the data directory.
    """
    # First, check if the video_name is already a full path
    if os.path.isfile(video_name):
        return video_name
    return resource_path(os.path.join('data', video_name))