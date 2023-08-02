from zipfile import ZipFile
import tensorflow as tf
import math
import numpy as np
import os

#limit ram to run multiple algorithms at once
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
# Optimizations
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Data treatement
if not os.path.exists("jena_climate_2009_2016.csv"):
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

# take only temperature each sample is 10 minutes appart
# I'm using temperature this is an auto regressive model
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
# 6 * 4 is 4 hours since we have 1 measure point every 10 minutes that is 240 minutes total which is 4 hours
# 6 * 2 is using the previous 2 hours to predict the next 2 hours
lag = 6*4
train_X = train_dataset[:-lag]
train_Y = train_dataset[lag:]

val_X = val_dataset[:-lag]
val_Y = val_dataset[lag:]

test_X = test_dataset[:-lag]
test_Y = test_dataset[lag:]

# for lag 10
# model
# [0.1884603053331375, 0.36764955520629883]
# model = tf.keras.Sequential([
#     tf.keras.Input(shape=(lag, 1)),
#     tf.keras.layers.LSTM(units=32, return_sequences=True, recurrent_dropout=0.25),
#     tf.keras.layers.LSTM(units=32, return_sequences=True, recurrent_dropout=0.25),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=lag, return_sequences=True, recurrent_dropout=0.25, unroll=True)),
#     tf.keras.layers.Dense(1)
# ])

# unroll all layers reduced from 65 sec per epoch to 21 with 256/128/32 batch size
# with batch size 512 takes 12 sec per epoch but ends at it 13 with
# Evaluation metrics best model's weights: [0.20584887266159058, 0.37631678581237793]
# and  loss: 0.3436 - mean_absolute_error: 0.4538 - val_loss: 0.3323 - val_mean_absolute_error: 0.4774 - lr: 0.0100
# which is worst
# with a batch of 256 the lr decay kicks in at epoch 16 the training does not end at epoch 13

# no bidirectional
# tree layers
# Evaluation metrics last model: [0.08787481486797333, 0.21113896369934082]
# model = tf.keras.Sequential([
#     tf.keras.Input(shape=(lag, 1)),
#     tf.keras.layers.LSTM(units=32, return_sequences=True, recurrent_dropout=0.25, unroll=True),
#     tf.keras.layers.LSTM(units=32, return_sequences=True, recurrent_dropout=0.25, unroll=True),
#     tf.keras.layers.LSTM(units=lag, return_sequences=True, recurrent_dropout=0.25, unroll=True),
#     tf.keras.layers.Dense(1)
# ])

# no bidirectional
#start 1537
# Evaluation metrics best model's weights: [0.08311255276203156, 0.20944900810718536]
# 13 sec per epoch
# loss: 0.2489 - mean_absolute_error: 0.3731 - val_loss: 0.1974 - val_mean_absolute_error: 0.3184 - lr: 4.0762e-04
# 335 epochs
# model = tf.keras.Sequential([
#     tf.keras.Input(shape=(lag, 1)),
#     tf.keras.layers.LSTM(units=16, return_sequences=True, recurrent_dropout=0.25, unroll=True),
#     # tf.keras.layers.LSTM(units=32, return_sequences=True, recurrent_dropout=0.25, unroll=True),
#     tf.keras.layers.LSTM(units=lag, return_sequences=True, recurrent_dropout=0.25, unroll=True),
#     tf.keras.layers.Dense(1)
# ])

# Evaluation metrics best model's weights: [0.08175995200872421, 0.20465394854545593]
# loss: 0.2497 - mean_absolute_error: 0.3735 - val_loss: 0.1969 - val_mean_absolute_error: 0.3180 - lr: 4.1172e-04
# model = tf.keras.Sequential([
#     tf.keras.Input(shape=(lag, 1)),
#     tf.keras.layers.LSTM(units=8, return_sequences=True, recurrent_dropout=0.25, unroll=True),
#     tf.keras.layers.LSTM(units=lag, return_sequences=True, recurrent_dropout=0.25, unroll=True),
#     tf.keras.layers.Dense(1)
# ])

#for lag 6*4 4 hours
# model = tf.keras.Sequential([
#     tf.keras.Input(shape=(lag, 1)),
#     tf.keras.layers.LSTM(units=32, return_sequences=True, recurrent_dropout=0.25, unroll=True),
#     tf.keras.layers.LSTM(units=lag, return_sequences=True, recurrent_dropout=0.25, unroll=True),
#     tf.keras.layers.Dense(1)
# ])

# at 19 epochs
# 7s 6ms/step - loss: 0.0084 - mean_absolute_error: 0.0648 - val_loss: 0.0036 - val_mean_absolute_error: 0.0438 - lr: 0.0096
# [0.003766818204894662, 0.04618332162499428]
model = tf.keras.Sequential([
    tf.keras.Input(shape=(lag, 1)),
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation="relu"),
    tf.keras.layers.Conv1D(filters=10, kernel_size=3, activation="relu"),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(10)
])

model.summary()

# Create datasets
train_dataset = tf.keras.utils.timeseries_dataset_from_array(
    train_X,
    train_Y,
    sequence_length=lag,
    sequence_stride=1,
    batch_size=256,
    # sampling_rate=6
)

val_dataset = tf.keras.utils.timeseries_dataset_from_array(
    val_X,
    val_Y,
    sequence_length=lag,
    sequence_stride=1,
    batch_size=128,
    # sampling_rate=6
)

test_dataset = tf.keras.utils.timeseries_dataset_from_array(
    test_X,
    test_Y,
    sequence_length=lag,
    sequence_stride=1,
    batch_size=32,
    # sampling_rate=6
)

# compile model
# Define optimizer and loss
model.compile(
    loss=tf.keras.losses.mean_squared_error,
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
    metrics=[tf.keras.metrics.mean_absolute_error],
    jit_compile=True
)


# Callbacks
def scheduler(epoch, lr):
    if epoch < 15:
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
