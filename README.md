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

Download model weight file from git lfs. Reference the MODEL_PATH constant in sharkeye_app.py
```
git lfs fetch -I model_weights/runs-detect-train-weights-best.pt
git lfs pull
```

Start App
```
python src/sharkeye.py
```

# PyInstaller app
Create an Executable App
```
pyinstaller SharkEye.spec --noconfirm
```
