import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

X_test = np.load("data/processed/X_test.npy")
y_test = np.load("data/processed/y_test.npy")

with open("data/processed/label_classes.txt") as f:
    class_names = [line.strip() for line in f.readlines()]

model = load_model("models/best_model_resnet.keras")
X_test_pp = tf.keras.applications.resnet50.preprocess_input(X_test * 255.0)
preds = model.predict(X_test_pp, verbose=0)
y_pred = np.argmax(preds, axis=1)
confidences = np.max(preds, axis=1)

print("Best correct, high-confidence example per class:\n")
for cls in np.unique(y_test):
    mask = (y_test == cls) & (y_pred == cls)
    if mask.sum() == 0:
        print(f"{class_names[cls]}: no correct predictions found")
        continue
    idx_in_class = np.where(mask)[0]
    best_idx = idx_in_class[np.argmax(confidences[idx_in_class])]
    print(f"{class_names[cls]}: index {best_idx}, confidence {confidences[best_idx]*100:.1f}%")