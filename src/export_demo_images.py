import numpy as np
from PIL import Image
import os

DATA_DIR = "data/processed"
OUTPUT_DIR = "demo_images"

# Best correct, high-confidence example per class (from find_good_demo_examples.py)
SAMPLES_TO_EXPORT = [
    (296, "akiec_example"),
    (508, "bcc_example"),
    (784, "bkl_example"),
    (119, "df_example"),
    (426, "melanoma_example"),
    (3, "nv_benign_example"),
    (74, "vasc_example"),
]

def main():
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for idx, name in SAMPLES_TO_EXPORT:
        img_array = (X_test[idx] * 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        path = os.path.join(OUTPUT_DIR, f"{name}.png")
        img.save(path)
        print(f"Saved {path} (true label index: {y_test[idx]})")

if __name__ == "__main__":
    main()