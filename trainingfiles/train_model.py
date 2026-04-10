import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os


IMG_SIZE = (224, 224)
BATCH = 32
SEED = 13
train_dir = "dataset/train"
val_dir = "dataset/val"
test_dir = "dataset/test"


train_ds = keras.preprocessing.image_dataset_from_directory(
    train_dir, image_size=IMG_SIZE, batch_size=BATCH, label_mode="int", shuffle=True, seed=SEED
)
val_ds = keras.preprocessing.image_dataset_from_directory(
    val_dir, image_size=IMG_SIZE, batch_size=BATCH, label_mode="int", shuffle=True, seed=SEED
)
test_ds = keras.preprocessing.image_dataset_from_directory(
    test_dir, image_size=IMG_SIZE, batch_size=BATCH, label_mode="int", shuffle=False
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)


base_model = tf.keras.applications.EfficientNetB0(
    include_top=False, input_shape=IMG_SIZE + (3,), weights="imagenet", pooling="avg"
)
base_model.trainable = False

inputs = keras.Input(shape=IMG_SIZE + (3,))
x = tf.keras.applications.efficientnet.preprocess_input(inputs)
x = base_model(x, training=False)
x = layers.Dropout(0.4)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs, outputs)
model.compile(optimizer=keras.optimizers.Adam(1e-4),
              loss="binary_crossentropy",
              metrics=["accuracy", keras.metrics.AUC(name="auc")])

model.summary()


callbacks = [
    keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True, monitor="val_auc", mode="max"),
    keras.callbacks.EarlyStopping(monitor="val_auc", patience=5, restore_best_weights=True),
]

history = model.fit(train_ds, validation_data=val_ds, epochs=4, callbacks=callbacks)


test_loss, test_acc, test_auc = model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc:.4f}, AUC: {test_auc:.4f}")

model.save("saved_deepfake_detector")
print(" Model saved successfully!")
