from pathlib import Path
import shutil

targets = [
    Path("data/processed"),
    Path("data/raw/tcga_prad/centers"),
]

for target in targets:
    if target.exists():
        shutil.rmtree(target)

# Recreate expected directories so downstream scripts do not fail
Path("data/processed").mkdir(parents=True, exist_ok=True)
Path("data/raw/tcga_prad/centers").mkdir(parents=True, exist_ok=True)

print("Pipeline outputs removed.")