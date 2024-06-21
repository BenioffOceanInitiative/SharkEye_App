# Setup
python -m venv ocean
source ocean/bin/activate
pip install -r requirements.txt
python src/sharkeye.py

# PyInstaller app
pyinstaller src/sharkeye.py --clean --noconfirm --log-level ERROR -n SharkEye --add-data '/Users/<username>/anaconda3/envs/ocean/lib/python3.10/site-packages/ultralytics/cfg':'ultralytics/cfg' --add-data 'model_weights/best.pt:model_weights' -w