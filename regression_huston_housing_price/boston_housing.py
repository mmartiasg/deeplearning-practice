import tensorflow as tf
import numpy as np
import math

# limit ram
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=256)])

# Optimizations
# GPU
tf.config.optimizer.set_jit(True)
# CPU?
tf.function(jit_compile=True)
# Mixed precision
# not compatible with trained weights of the efficient net
tf.keras.mixed_precision.set_global_policy('mixed_float16')

(X, Y), (X_Test, Y_Test) = tf.keras.datasets.boston_housing.load_data()

print("Train and label samples shape")
print(X.shape)
print(X[0])
print(Y.shape)
print(Y[0])
print(f"Test shape: X: {X_Test.shape} - Y : {Y_Test.shape}")

# Split train and validation
m = X.shape[0]
n_train = math.ceil(m*0.7)

# Initial shuffle to avoid bias by splitting
np.random.RandomState(42).shuffle(X)
np.random.RandomState(42).shuffle(Y)

X_Train = X[:n_train]
Y_Train = Y[:n_train]
X_Val = X[n_train:]
Y_Val = X[n_train:]
print(f"Train shape{X_Train.shape} - Val shape: {X_Val.shape}|Train label shape: {Y_Train.shape} - Val label shape: {Y_Val.shape}")
# End Split

# Normalize Values!
mean = X_Train.mean()
std = X_Train.std()

X_Train -= mean
X_Train /= std

X_Val -= mean
X_Val /= std

X_Test -= mean
X_Test /= std
# End normalization

# Prepare datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_Train, Y_Train)).shuffle(256).batch(64).prefetch(tf.data.AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices((X_Val, Y_Val)).batch(32).prefetch(tf.data.AUTOTUNE).cache()
test_dataset = tf.data.Dataset.from_tensor_slices((X_Test, Y_Test)).batch(8).prefetch(tf.data.AUTOTUNE)

# Build model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(13,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1)
])

# This is a single variable regression problem thus the output layer will be a linear activation of 1 unit
# Compile
# Loss is just the squared distance and the metric is the absolute error
model.compile(loss="mean_squared_error",
              metrics=["mean_absolute_error"],
              optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-2))

# Callbacks this is allways the same I will use:
# model checkpoint, early stop time is of the essence and learning rate decrease (optional)
early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="checkpoint",
                                                      save_weights_only=True,
                                                      save_only_best=True,
                                                      verbose=1)

model.fit(train_dataset, validation_data=val_dataset, epochs=500, callbacks=[model_checkpoint, early_stop])

# Load weights
model.load_weights("checkpoint")
print(f"Test dataset evaluation: {model.evaluate(test_dataset)}")

model.save(filepath="huston_housing_model.h5", save_format="h5")




