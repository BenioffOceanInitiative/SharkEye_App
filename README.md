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

# Installing on macOS
The macOS build ships as a `.dmg`. Open it and drag **SharkEye** into your
**Applications** folder.

Because the app is not yet Apple-notarized, macOS quarantines it on first download.
On older macOS you can right-click the app → **Open**. On macOS 15 (Sequoia) that option
is gone, so run this once to clear the quarantine flag:
```
xattr -dr com.apple.quarantine /Applications/SharkEye.app
```
The app will then launch normally. (Notarizing the build with an Apple Developer ID would
remove this step for everyone — see the distribution notes in the build plan.)
