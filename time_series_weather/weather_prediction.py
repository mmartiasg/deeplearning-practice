from zipfile import ZipFile
import tensorflow as tf
import math
import numpy as np

# optimizer
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])

# Optimizations
# GPU
tf.config.optimizer.set_jit(True)
# CPU?
tf.function(jit_compile=True)
# Mixed precision
# not compatible with trained weights of the efficient net
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Data treatement
uri = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
zip_path = tf.keras.utils.get_file(origin=uri, fname="jena_climate_2009_2016.csv.zip")
zip_file = ZipFile(zip_path)
zip_file.extractall()
csv_path = "jena_climate_2009_2016.csv"

data = []

with open("jena_climate_2009_2016.csv") as f:
    for index, row in enumerate(f):
        if index > 0:
            line = [float(c) for c in row.split(sep=",")[1:]]
            data.append(line)
        else:
            headers = row.split(sep=",")[1:]

dataset = np.array(data, dtype="float32")

# take only temperature
dataset = dataset[:, 1]

print(dataset[:5])

# We get 10 measurements and return 10 next measures
# Split by time
m = dataset.shape[0]
train_n = math.ceil(m*0.7)
val_n = math.ceil(m*0.2)

train_dataset = dataset[:train_n]
val_dataset = dataset[train_n:train_n+val_n]
test_dataset = dataset[train_n+val_n:]

# normalize
mean = train_dataset.mean()
std = train_dataset.std()

train_dataset -= mean
train_dataset /= std

val_dataset -= mean
val_dataset /= std

test_dataset -= mean
test_dataset /= std

# 10 previous measures
lag = 10
train_X = train_dataset[:-lag]
train_Y = train_dataset[lag:]

val_X = val_dataset[:-lag]
val_Y = val_dataset[lag:]

test_X = test_dataset[:-lag]
test_Y = test_dataset[lag:]

# model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(lag, 1)),
    tf.keras.layers.LSTM(units=lag, return_sequences=True),
    tf.keras.layers.Dense(1, activation="relu")
])

model.summary()

# Create datasets
train_dataset = tf.keras.utils.timeseries_dataset_from_array(
    train_X,
    train_Y,
    sequence_length=lag,
    sequence_stride=1,
    batch_size=256
)

val_dataset = tf.keras.utils.timeseries_dataset_from_array(
    val_X,
    val_Y,
    sequence_length=lag,
    sequence_stride=1,
    batch_size=64
)

test_dataset = tf.keras.utils.timeseries_dataset_from_array(
    test_X,
    test_Y,
    sequence_length=lag,
    sequence_stride=1,
    batch_size=16
)

# compile model
# Define optimizer and loss
model.compile(
    loss=tf.keras.losses.mean_squared_error,
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    metrics=[tf.keras.metrics.mean_absolute_error]
)


# Callbacks
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.01)


reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)
early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
save_best_model = tf.keras.callbacks.ModelCheckpoint(filepath="checkpoint_model.h5", save_only_best=True,
                                                     save_weights_only=True, verbose=1)

# Train model
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=500,
    callbacks=[early_stop_callback, save_best_model, reduce_lr]
)

# evaluate model best model
print(f"Evaluation metrics last model: {model.evaluate(test_dataset)}")

model.load_weights("checkpoint_model.h5")
print(f"Evaluation metrics best model's weights: {model.evaluate(test_dataset)}")

model.save(filepath="jena_climate_model.h5", save_format="h5")
