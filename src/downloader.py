import kaggle
import kagglehub
from pathlib import Path
import shutil
import uuid
import cv2

image_exts = {".jpg", ".jpeg", ".png", ".webp"}
CLASSES = ["king", "queen", "rook", "bishop", "knight", "pawn"]


# Helper function to infer class from folder name
def infer_class(folder_name: str):
    name = folder_name.lower()
    for cls in CLASSES:
        if cls in name:
            return cls
    return None

# Function to merge datasets
def merge_dataset(src_root: Path, dst_root: Path):
    for folder in src_root.iterdir():
        if not folder.is_dir():
            continue

        cls = infer_class(folder.name)
        if cls is None:
            continue  # skip unrelated folders

        dst_cls = dst_root / cls
        dst_cls.mkdir(parents=True, exist_ok=True)

        for img in folder.iterdir():
            if img.suffix.lower() not in image_exts:
                continue

            new_name = f"{img.stem}_{uuid.uuid4().hex[:8]}{img.suffix}"
            shutil.copy2(img, dst_cls / new_name)

def download_dataset():
    """
    Downloads the dataset from Kaggle Hub.
    Note: this requires a Kaggle API key to be set up, which can be gained from a Kaggle account settings page.
    If a Kaggle.json file is downloaded, it should be placed in ~/.kaggle/
    """

    path_one = kagglehub.dataset_download("anshulmehtakaggl/chess-pieces-detection-images-dataset")
    path_two = kagglehub.dataset_download("niteshfre/chessman-image-dataset")

    data_root = Path("../data/merged_chess_dataset")

    if data_root.exists():
        shutil.rmtree(data_root)
    data_root.mkdir(parents=True)
    path_one = Path(path_one)
    path_two = Path(f"{path_two}/Chessman-image-dataset/Chess")

    print(f"Dataset one downloaded to: {path_one}")
    print(f"Dataset two downloaded to: {path_two}")

    merge_dataset(path_one, data_root)
    merge_dataset(path_two, data_root)

    print("data/\n\tmerged_chess_dataset/")
    for cls in sorted(p for p in data_root.iterdir() if p.is_dir()):
        print(f"\t\t{cls.name}/  ({len(list(cls.glob('*')))} images)")


if __name__ == "__main__":
    download_dataset()