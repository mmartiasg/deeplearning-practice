import tensorflow_datasets as tfds
import tensorflow as tf
import math
# Smartwatch
import numpy as np

# limit ram to run multiple algorithms at once
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(
gpus[0],
[tf.config.LogicalDeviceConfiguration(memory_limit=1024)])

# Beans dataset
with tf.device("CPU") as d:
    X, Y = tfds.as_numpy(tfds.load(
        'smartwatch_gestures',
        split='train',
        batch_size=-1,
        as_supervised=True,
    ))

features = [str(f) for f in X.keys()]

features_array = []
for f in features:
    features_array.append(X[f])

X_dataset = np.column_stack(features_array)

# Batch Timesteps features
# X_dataset = np.array(X_dataset, dtype="float32").reshape((-1, 51))

print(f"Dataset shape: {X_dataset.shape}")

CLASSES = 20

# Split train val test
N = X_dataset.shape[0]
n_train = math.ceil(N * 0.8)
n_val = math.ceil(N * 0.15)

X_Train = X_dataset[:n_train]
Y_Train = Y[:n_train]

X_Val = X_dataset[n_train:]
Y_Val = Y[n_train:]

X_Test = X_dataset[n_train + n_val:]
Y_Test = Y[n_train + n_val:]

print(f"Train size:{X_Train.shape} | Val size: {X_Val.shape} | Test size:{X_Test.shape}")

# prepare dataset
train_dataset = tf.keras.utils.timeseries_dataset_from_array(
    X_Train,
    Y_Train,
    sequence_length=6,
    sequence_stride=1,
    batch_size=128,
)

val_dataset = tf.keras.utils.timeseries_dataset_from_array(
    X_Val,
    Y_Val,
    sequence_length=6,
    sequence_stride=1,
    batch_size=64
)

test_dataset = tf.keras.utils.timeseries_dataset_from_array(
    X_Test,
    Y_Test,
    sequence_length=6,
    sequence_stride=1,
    batch_size=64
)

# model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=128, recurrent_dropout=0.2, return_sequences=True),
    tf.keras.layers.LSTM(units=128, recurrent_dropout=0.2, return_sequences=False),
    tf.keras.layers.Dense(units=128, activation="relu"),
    tf.keras.layers.Dropout(rate=0.4),
    tf.keras.layers.Dense(units=CLASSES, activation="softmax")
]
)

model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              # TODO: Check half of the classes are underrepresented?? with k=10 I get over 70% accuracy
              metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10)],
              jit_compile=True)

early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="temporal_checkpoint",
                                                      save_weights_only=True,
                                                      save_only_best=True,
                                                      verbose=1)

model.fit(train_dataset, validation_data=val_dataset, epochs=500, callbacks=[early_stop, model_checkpoint])
model.load_weights("temporal_checkpoint")

print(f"Model evaluate on test: {model.evaluate(test_dataset)}")

model.save(filepath="smart_watch_gestures.h5", save_format="h5")
