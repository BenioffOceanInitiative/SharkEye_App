# hook-ultralytics.py

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect all submodules
hiddenimports = collect_submodules('ultralytics')

# Collect all data files
datas = collect_data_files('ultralytics', include_py_files=True)