import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import math

# limit ram to run multiple algorithms at once
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=256)])

# Optimizations
tf.keras.mixed_precision.set_global_policy('mixed_float16')

with tf.device("/CPU") as device:
    X, Y = tfds.as_numpy(tfds.load(
        'wine_quality',
        split='train',
        batch_size=-1,
        as_supervised=True,
    ))

features = [str(f) for f in X.keys()]

features_array = []
for f in features:
    features_array.append(X[f])

X_dataset = np.column_stack(features_array)

# Split data
N = X_dataset.shape[0]
n_train = math.ceil(N * 0.8)

np.random.RandomState(42).shuffle(X_dataset)
np.random.RandomState(42).shuffle(Y)

X_train = X_dataset[:n_train]
Y_train = Y[:n_train]

X_val = X_dataset[:n_train]
Y_val = Y[:n_train]

# normalize data
mean = X_train.mean()
std = X_train.std()

X_train -= mean
X_train /= std

X_val -= mean
X_val /= std

# prepare datasets
train_datasets = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(128).batch(128).prefetch(tf.data.AUTOTUNE)
val_datasets = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(32).prefetch(tf.data.AUTOTUNE)

# Create model
X_inputs = tf.keras.Input(shape=(len(features),))
X = tf.keras.layers.Dense(units=256, activation="relu")(X_inputs)
X = tf.keras.layers.Dense(units=128, activation="relu")(X)
X = tf.keras.layers.Dense(units=64, activation="relu")(X)
X = tf.linalg.l2_normalize(X)
X_output = tf.keras.layers.Dense(units=1)(X)
model = tf.keras.Model(inputs=[X_inputs], outputs=[X_output])

# model = tf.keras.Sequential([
#     # tf.keras.Input(shape=(13,)),
#     tf.keras.layers.Dense(256, activation="relu"),
#     tf.keras.layers.Dense(128, activation="relu"),
#     tf.keras.layers.Dense(64, activation="relu"),
#     tf.keras.layers.Dense(1)
# ])


# compile model
model.compile(loss=tf.keras.losses.mean_squared_error,
              optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              metrics="MAE",
              jit_compile=True)

# callbacks
early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="temporal_checkpoint",
                                                      save_weights_only=True,
                                                      save_only_best=True,
                                                      verbose=1)

model.fit(train_datasets, validation_data=val_datasets, epochs=500, callbacks=[early_stop, model_checkpoint])

model.load_weights("temporal_checkpoint")

model.save(filepath="wine_quality.h5", save_format="h5")

