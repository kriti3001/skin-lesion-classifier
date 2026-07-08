import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
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

# Augmentation as a Keras layer pipeline (only active during training)
augmenter = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.11),      # ~40 degrees
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomBrightness(0.15, value_range=(0.0, 1.0)),
])

def make_dataset(X, y, batch_size, training=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(buffer_size=len(X), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    if training:
        ds = ds.map(lambda x, y: (augmenter(x, training=True), y),
                     num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.repeat()  # critical: keeps generating data across all epochs
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

class MacroF1Checkpoint(Callback):
    def __init__(self, X_val, y_val, filepath):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.filepath = filepath
        self.best_f1 = -1.0

    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.predict(self.X_val, verbose=0)
        y_pred = np.argmax(preds, axis=1)
        macro_f1 = f1_score(self.y_val, y_pred, average="macro")
        print(f"  val_macro_f1: {macro_f1:.4f}")
        if macro_f1 > self.best_f1:
            self.best_f1 = macro_f1
            self.model.save(self.filepath)
            print(f"  New best macro-F1 ({macro_f1:.4f}). Saved model.")

def build_model():
    base_model = MobileNetV2(input_shape=IMG_SIZE, include_top=False, weights="imagenet")
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

def main():
    print("Loading data...")
    X_train, y_train, X_val, y_val = load_data()

    print("Oversampling minority classes...")
    print("Before:", {int(c): int((y_train == c).sum()) for c in np.unique(y_train)})
    X_train, y_train = oversample(X_train, y_train, TARGET_SAMPLES_PER_CLASS)
    print("After:", {int(c): int((y_train == c).sum()) for c in np.unique(y_train)})

    print("Computing light class weights (mild correction on top of oversampling)...")
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    print("Class weights:", class_weight_dict)

    steps_per_epoch = len(X_train) // BATCH_SIZE
    train_ds = make_dataset(X_train, y_train, BATCH_SIZE, training=True)
    val_ds = make_dataset(X_val, y_val, BATCH_SIZE, training=False)

    print("Building model...")
    model, base_model = build_model()
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    os.makedirs(MODEL_DIR, exist_ok=True)
    checkpoint_path = os.path.join(MODEL_DIR, "best_model.keras")
    f1_checkpoint = MacroF1Checkpoint(X_val, y_val, checkpoint_path)

    print("\n=== Phase 1: Training classifier head (frozen backbone) ===")
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

    print("\n=== Phase 2: Fine-tuning MobileNetV2 (last 60 layers unfrozen) ===")
    base_model.trainable = True
    for layer in base_model.layers[:-60]:
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

    model.save(os.path.join(MODEL_DIR, "final_model.keras"))
    print(f"\nTraining complete. Best macro-F1 during training: {f1_checkpoint.best_f1:.4f}")
    print("Best model (by macro-F1) saved to", checkpoint_path)

if __name__ == "__main__":
    main()