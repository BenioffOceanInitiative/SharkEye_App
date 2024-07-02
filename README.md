# Setup
Setup a Virtual Environment
```
python -m venv ocean
```

Activate Virtual Environment
```
source ocean/bin/activate
```

Install Python Libraries
```
pip install -r requirements.txt
```

Start App
```
python src/main.py
```

# PyInstaller app
Create an Executable App
```
pyinstaller src/sharkeye.py --clean --noconfirm --log-level ERROR -n SharkEye --add-data '/Users/<username>/anaconda3/envs/ocean/lib/python3.10/site-packages/ultralytics/cfg':'ultralytics/cfg' --add-data 'model_weights/best.pt:model_weights' -w
```
