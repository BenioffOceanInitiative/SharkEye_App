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

Install Model Weights
```
python model_weights/download_models_from_gcs.py
```

Start App
```
python src/sharkeye_app.py
```

# PyInstaller app
Create an Executable App
```
pyinstaller SharkEye.spec --noconfirm
```
