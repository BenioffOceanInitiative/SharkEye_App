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
python src/sharkeye.py
```

# PyInstaller app
Create an Executable App
```
pyinstaller SharkEye.spec --noconfirm
```

# For the segmentation model
Download the `sam_vit_b_01ec64.pth` model from the [segment-anything GitHub](https://github.com/facebookresearch/segment-anything/tree/main?tab=readme-ov-file#model-checkpoints) repo and place the model in `src/segmentation` in order for the segmentation model to run properly.