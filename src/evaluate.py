import numpy as np
import os
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "data/processed"
MODEL_PATH = "models/best_model.keras"

def main():
    print("Loading test data...")
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    with open(os.path.join(DATA_DIR, "label_classes.txt")) as f:
        class_names = [line.strip() for line in f.readlines()]

    print("Loading model...")
    model = load_model(MODEL_PATH)

    print("Predicting on test set...")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\nClassification Report:\n")
    report = classification_report(y_test, y_pred, target_names=class_names, digits=3)
    print(report)

    # Save report to file
    with open("models/evaluation_report.txt", "w") as f:
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Test Set")
    plt.tight_layout()
    plt.savefig("models/confusion_matrix.png")
    print("\nSaved evaluation_report.txt and confusion_matrix.png to models/")

if __name__ == "__main__":
    main()