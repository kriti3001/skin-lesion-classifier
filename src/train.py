import numpy as np
import os
import argparse
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0, ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score

DATA_DIR = "data/processed"
MODEL_DIR = "models"
IMG_SIZE = (224, 224, 3)
NUM_CLASSES = 7
BATCH_SIZE = 32
INITIAL_EPOCHS = 25
FINE_TUNE_EPOCHS = 25
TARGET_SAMPLES_PER_CLASS = 1200

BACKBONES = {
    "mobilenet": {
        "fn": MobileNetV2,
        "preprocess": tf.keras.applications.mobilenet_v2.preprocess_input,
        "fine_tune_layers": 60,
    },
    "efficientnet": {
        "fn": EfficientNetB0,
        "preprocess": tf.keras.applications.efficientnet.preprocess_input,
        "fine_tune_layers": 40,
    },
    "resnet": {
        "fn": ResNet50,
        "preprocess": tf.keras.applications.resnet50.preprocess_input,
        "fine_tune_layers": 30,
    },
}

def load_data():
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    return X_train, y_train, X_val, y_val

def oversample(X, y, target_count):
    X_list, y_list = [X], [y]
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        n_have = len(idx)
        if n_have < target_count:
            n_needed = target_count - n_have
            extra_idx = np.random.choice(idx, size=n_needed, replace=True)
            X_list.append(X[extra_idx])
            y_list.append(y[extra_idx])
    X_new = np.concatenate(X_list, axis=0)
    y_new = np.concatenate(y_list, axis=0)
    perm = np.random.permutation(len(y_new))
    return X_new[perm], y_new[perm]

augmenter = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.11),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomBrightness(0.15, value_range=(0.0, 1.0)),
])

def make_dataset(X, y, batch_size, preprocess_fn, training=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(buffer_size=len(X), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    if training:
        ds = ds.map(lambda x, y: (augmenter(x, training=True), y),
                     num_parallel_calls=tf.data.AUTOTUNE)
    # Note: our arrays are already scaled to [0,1] from preprocessing.
    # Each backbone's preprocess_input expects a specific input convention,
    # so we rescale back to [0,255] before applying it.
    ds = ds.map(lambda x, y: (preprocess_fn(x * 255.0), y),
                 num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.repeat()
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

class MacroF1Checkpoint(Callback):
    def __init__(self, X_val, y_val, preprocess_fn, filepath):
        super().__init__()
        self.X_val_raw = X_val
        self.y_val = y_val
        self.preprocess_fn = preprocess_fn
        self.filepath = filepath
        self.best_f1 = -1.0

    def on_epoch_end(self, epoch, logs=None):
        X_val_pp = self.preprocess_fn(self.X_val_raw * 255.0)
        preds = self.model.predict(X_val_pp, verbose=0)
        y_pred = np.argmax(preds, axis=1)
        macro_f1 = f1_score(self.y_val, y_pred, average="macro")
        print(f"  val_macro_f1: {macro_f1:.4f}")
        if macro_f1 > self.best_f1:
            self.best_f1 = macro_f1
            self.model.save(self.filepath)
            print(f"  New best macro-F1 ({macro_f1:.4f}). Saved model.")

def build_model(backbone_name):
    config = BACKBONES[backbone_name]
    base_model = config["fn"](input_shape=IMG_SIZE, include_top=False, weights="imagenet")
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    predictions = Dense(NUM_CLASSES, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model

def main(backbone_name):
    config = BACKBONES[backbone_name]
    preprocess_fn = config["preprocess"]
    fine_tune_layers = config["fine_tune_layers"]

    print(f"\n{'='*60}\nTraining backbone: {backbone_name}\n{'='*60}\n")

    print("Loading data...")
    X_train, y_train, X_val, y_val = load_data()

    print("Oversampling minority classes...")
    X_train, y_train = oversample(X_train, y_train, TARGET_SAMPLES_PER_CLASS)
    print("Class counts after oversampling:", {int(c): int((y_train == c).sum()) for c in np.unique(y_train)})

    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    steps_per_epoch = len(X_train) // BATCH_SIZE
    train_ds = make_dataset(X_train, y_train, BATCH_SIZE, preprocess_fn, training=True)
    val_ds = make_dataset(X_val, y_val, BATCH_SIZE, preprocess_fn, training=False)

    print("Building model...")
    model, base_model = build_model(backbone_name)
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    os.makedirs(MODEL_DIR, exist_ok=True)
    checkpoint_path = os.path.join(MODEL_DIR, f"best_model_{backbone_name}.keras")
    f1_checkpoint = MacroF1Checkpoint(X_val, y_val, preprocess_fn, checkpoint_path)

    print(f"\n=== Phase 1: Training classifier head ({backbone_name}, frozen backbone) ===")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=INITIAL_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        class_weight=class_weight_dict,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
            f1_checkpoint,
        ],
    )

    print(f"\n=== Phase 2: Fine-tuning {backbone_name} (last {fine_tune_layers} layers unfrozen) ===")
    base_model.trainable = True
    for layer in base_model.layers[:-fine_tune_layers]:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINE_TUNE_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        class_weight=class_weight_dict,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
            f1_checkpoint,
        ],
    )

    model.save(os.path.join(MODEL_DIR, f"final_model_{backbone_name}.keras"))
    print(f"\n{backbone_name} training complete. Best macro-F1: {f1_checkpoint.best_f1:.4f}")
    print(f"Best model saved to {checkpoint_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="mobilenet",
                         choices=list(BACKBONES.keys()),
                         help="Which backbone to train: mobilenet, efficientnet, or resnet")
    args = parser.parse_args()
    main(args.backbone)