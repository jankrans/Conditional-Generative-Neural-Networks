from pathlib import Path
import os

for path in Path().glob('**/*.ipynb'):
    if not '.ipynb_checkpoints' in str(path):
        os.system(f"jupyter nbconvert --clear-output --inplace {path}")
