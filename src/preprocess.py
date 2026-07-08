import pandas as pd
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Paths
DATA_DIR = "data"
METADATA_PATH = os.path.join(DATA_DIR, "HAM10000_metadata.csv")
IMAGE_DIRS = [
    os.path.join(DATA_DIR, "HAM10000_images_part_1"),
    os.path.join(DATA_DIR, "HAM10000_images_part_2"),
]
IMG_SIZE = (224, 224)
OUTPUT_DIR = "data/processed"

def build_image_path_map():
    """Map each image_id to its actual file path across both folders."""
    path_map = {}
    for folder in IMAGE_DIRS:
        for fname in os.listdir(folder):
            if fname.endswith(".jpg"):
                image_id = fname.replace(".jpg", "")
                path_map[image_id] = os.path.join(folder, fname)
    return path_map

def load_and_preprocess():
    print("Loading metadata...")
    df = pd.read_csv(METADATA_PATH)

    print("Mapping image paths...")
    path_map = build_image_path_map()
    df["path"] = df["image_id"].map(path_map)

    # Drop any rows where the image file wasn't found
    missing = df["path"].isna().sum()
    if missing > 0:
        print(f"Warning: {missing} images not found, dropping those rows.")
        df = df.dropna(subset=["path"])

    print("Encoding labels...")
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["dx"])
    print("Class mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

    print(f"Loading and resizing {len(df)} images to {IMG_SIZE}...")
    images = []
    for i, path in enumerate(df["path"]):
        img = Image.open(path).convert("RGB").resize(IMG_SIZE)
        images.append(np.array(img))
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(df)}")

    X = np.array(images, dtype=np.float32) / 255.0  # normalize to [0,1]
    y = df["label"].values

    print("Splitting into train/val/test (70/15/15)...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(OUTPUT_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

    # Save label mapping for later use (e.g. in the Streamlit app)
    with open(os.path.join(OUTPUT_DIR, "label_classes.txt"), "w") as f:
        for cls in le.classes_:
            f.write(cls + "\n")

    print("Done. Saved processed arrays to", OUTPUT_DIR)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

if __name__ == "__main__":
    load_and_preprocess()