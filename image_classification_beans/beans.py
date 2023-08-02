import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

# limit ram to run multiple algorithms at once
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=2048)])

# Beans dataset
# 500 x 500 x 3 channels
with tf.device("CPU") as d:
    X_Train, Y_Train = tfds.as_numpy(tfds.load(
        'beans',
        split='train',
        batch_size=-1,
        as_supervised=True,
    ))
    X_Val, Y_Val = tfds.as_numpy(tfds.load(
        'beans',
        split='validation',
        batch_size=-1,
        as_supervised=True,
    ))
    X_test, Y_test = tfds.as_numpy(tfds.load(
        'beans',
        split='test',
        batch_size=-1,
        as_supervised=True,
    ))

CLASSES = 3
WIDTH=HEIGHT=128

# DataGen
datagen_train = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=45,
    brightness_range=(0.25, 1.25),
    width_shift_range=0.20,
    height_shift_range=0.20,
    horizontal_flip=True,
    vertical_flip=True)

datagen_val = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255
)

datagen_test = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255
)

# model
model = tf.keras.Sequential([
    tf.keras.layers.Resizing(height=HEIGHT, width=WIDTH),
    tf.keras.Input(shape=(WIDTH, HEIGHT, 3)),

    tf.keras.layers.Conv2D(64, 3, padding="valid"),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, 3, padding="valid"),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(256, 3, padding="valid"),
    tf.keras.layers.Activation("relu"),

    tf.keras.layers.GlobalMaxPool2D(),

    tf.keras.layers.Dense(4096, activation="relu"),
    tf.keras.layers.Dropout(rate=0.4),
    tf.keras.layers.Dense(4096, activation="relu"),
    tf.keras.layers.Dropout(rate=0.4),

    tf.keras.layers.Dense(CLASSES, activation="softmax")
])

# compile model
model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              metrics=[tf.keras.metrics.sparse_categorical_accuracy],
              jit_compile=True)

# Callbacks
early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="temporal_checkpoint",
                                                      save_weights_only=True,
                                                      save_only_best=True,
                                                      verbose=1)

model.fit(datagen_train.flow(X_Train, Y_Train, batch_size=32),
validation_data=datagen_val.flow(X_Val, Y_Val, batch_size=16),
epochs=500, callbacks=[model_checkpoint, early_stop])

model.load_weights("temporal_checkpoint")

model.save(filepath="wine_quality.h5", save_format="h5")

print(f"Eval on test: {model.evaluate(datagen_test.flow(X_test, Y_test, batch_size=16))}")
