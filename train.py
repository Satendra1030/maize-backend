import os
import tensorflow as tf

# ==========================================
# 1. CONFIGURATION & PATH SETUP
# ==========================================
# Path to your dataset folder containing class subdirectories 
# (e.g., dataset/maize and dataset/non_maize)
DATASET_DIR = "dataset"

if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(
        f"\n[ERROR] Directory '{DATASET_DIR}' was not found.\n"
        f"Please create the folder '{DATASET_DIR}' inside G:\\maize-backend\\ "
        f"and add subfolders for your classes (e.g., '{DATASET_DIR}/maize' and '{DATASET_DIR}/not_maize')."
    )

BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
EPOCHS = 10

# ==========================================
# 2. LOAD DATASET (Auto 80/20 Train/Val Split)
# ==========================================
print("Loading training dataset...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,  # Automatically reserves 20% for validation
    subset="training",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)

print("Loading validation dataset...")
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,  # Uses the remaining 20% for validation
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)

# Optimize memory and disk read speeds during training
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ==========================================
# 3. BUILD EFFICIENTNETB0 MODEL
# ==========================================
base_model = tf.keras.applications.EfficientNetB0(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
)
base_model.trainable = False  # Freeze ImageNet pre-trained weights

inputs = tf.keras.Input(shape=(224, 224, 3), name="input_image")
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# ==========================================
# 4. COMPILE & TRAIN
# ==========================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

print("\nStarting model training...")
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# ==========================================
# 5. SAVE MODEL
# ==========================================
os.makedirs("model", exist_ok=True)
SAVE_PATH = "model/gatekeeper_final.keras"
model.save(SAVE_PATH)
print(f"\nSUCCESS! Saved trained model to {SAVE_PATH}")