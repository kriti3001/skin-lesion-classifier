import numpy as np
import os
import argparse
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "data/processed"

def main(model_path, report_suffix):
    print("Loading test data...")
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    with open(os.path.join(DATA_DIR, "label_classes.txt")) as f:
        class_names = [line.strip() for line in f.readlines()]

    print("Loading model...")
    model = load_model(model_path)

    print("Predicting on test set...")
    if "efficientnet" in model_path:
        X_test_pp = tf.keras.applications.efficientnet.preprocess_input(X_test * 255.0)
    elif "resnet" in model_path:
        X_test_pp = tf.keras.applications.resnet50.preprocess_input(X_test * 255.0)
    else:
        X_test_pp = tf.keras.applications.mobilenet_v2.preprocess_input(X_test * 255.0)

    y_pred_probs = model.predict(X_test_pp)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\nClassification Report:\n")
    report = classification_report(y_test, y_pred, target_names=class_names, digits=3)
    print(report)

    # Save report to file
    with open(f"models/evaluation_report_{report_suffix}.txt", "w") as f:
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - Test Set ({report_suffix})")
    plt.tight_layout()
    plt.savefig(f"models/confusion_matrix_{report_suffix}.png")
    print(f"\nSaved evaluation_report_{report_suffix}.txt and confusion_matrix_{report_suffix}.png to models/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model file")
    parser.add_argument("--suffix", type=str, required=True, help="Suffix for output filenames")
    args = parser.parse_args()
    main(args.model, args.suffix)