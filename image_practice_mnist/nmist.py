import tensorflow as tf
import numpy as np
import math

# limit ram
gpus = tf.config.list_physical_devices('GPU')
# 1GB will do
tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])

# Optimization
tf.config.optimizer.set_jit(True)
# Mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

(X, Y), (X_Test, Y_Test) = tf.keras.datasets.mnist.load_data()

print(X.shape)
print(Y.shape)

X = X.astype("float16")
Y = Y.astype("float16")

X_Test = X_Test.astype("float16")
Y_Test = Y_Test.astype("float16")

# Normalize values
X = X * (1. / 255)
X_Test = X_Test * (1. / 255)

# Split dataset into train and validation
m = X.shape[0]
n_train = math.ceil(m * 0.8)
np.random.RandomState(42).shuffle(X)
np.random.RandomState(42).shuffle(Y)

X_Train = X[:n_train]
Y_Train = Y[:n_train]
X_Val = X[n_train:]
Y_Val = Y[n_train:]

train_data = tf.data.Dataset.from_tensor_slices((X_Train, Y_Train)).batch(128).prefetch(tf.data.AUTOTUNE)
val_data = tf.data.Dataset.from_tensor_slices((X_Val, Y_Val)).batch(64).prefetch(tf.data.AUTOTUNE)
test_data = tf.data.Dataset.from_tensor_slices((X_Test, Y_Test)).batch(64).prefetch(tf.data.AUTOTUNE)

# Create model
model = tf.keras.Sequential([tf.keras.Input(shape=(28, 28)),
                             tf.keras.layers.Flatten(),
                             tf.keras.layers.Dense(512, activation="relu"),
                             tf.keras.layers.Dense(512, activation="relu"),
                             tf.keras.layers.Dense(512, activation="relu"),
                             tf.keras.layers.Dropout(rate=0.4),
                             # from 0 to 9 are 10 values
                             tf.keras.layers.Dense(10, activation="softmax")])

# compile model
model.compile(loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
              metrics=["sparse_categorical_accuracy"])


def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


# Define callbacks
early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
save_best_model = tf.keras.callbacks.ModelCheckpoint(filepath="checkpoint", save_only_best=True,
                                                     save_weights_only=True, verbose=1)
reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)

# Train
model.fit(train_data,
          validation_data=val_data,
          epochs=500, callbacks=[early_stop_callback, save_best_model, reduce_lr],
          batch_size=128)

print(f"Evaluation metrics last model: {model.evaluate(test_data)}")

model.load_weights("best_model")
print(f"Evaluation metrics best model's weights: {model.evaluate(test_data)}")

model.save(filepath="nmist_model.h5", save_format="h5")
