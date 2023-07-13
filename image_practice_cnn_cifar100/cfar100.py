import tensorflow as tf
import numpy as np
import math

# limit ram
# gpus = tf.config.list_physical_devices('GPU')
# tf.config.set_logical_device_configuration(
    # gpus[0],
    # [tf.config.LogicalDeviceConfiguration(memory_limit=4196)])

# Optimizations
# GPU
# tf.config.optimizer.set_jit(True)
# CPU?
# tf.function(jit_compile=True)
# Mixed precision
# not compatible with trained weights of the efficient net
# tf.keras.mixed_precision.set_global_policy('mixed_float16')


(X, Y), (X_Test, Y_Test) = tf.keras.datasets.cifar100.load_data()

# Explore data
print(X.shape)
print(Y.shape)
print(Y[0])

# Set types to reduce memory load
X = X.astype("float16")
Y = Y.astype("float16")
X_Test = X_Test.astype("float16")
Y_Test = Y_Test.astype("float16")

# Normalize values
# this is done in a layer after the inputs
# X = X * (1./255)
# X_Test = X_Test * (1./255)

# Split dataset into train and validation
m = X.shape[0]
n_train = math.ceil(m * 0.8)
np.random.RandomState(42).shuffle(X)
np.random.RandomState(42).shuffle(Y)
X_Train = X[:n_train]
Y_Train = Y[:n_train]
X_Val = X[n_train:]
Y_Val = Y[n_train:]

print(f"Train size: {n_train} - Val size: {m-n_train}")
print(f"Test size: {X_Test.shape[0]}")


def augmentation(batch_image, batch_label):
    seed = tf.random.experimental.stateless_split((np.random.random_integers(0, 512),
                                                   np.random.random_integers(0, 512)),
                                                  num=1)[0, :]

    batch_image = tf.image.stateless_random_flip_up_down(batch_image, seed)
    batch_image = tf.image.stateless_random_brightness(batch_image, max_delta=0.95, seed=seed)
    batch_image = tf.image.stateless_random_contrast(batch_image, lower=0.1, upper=0.9, seed=seed)
    batch_image = tf.image.stateless_random_flip_left_right(batch_image, seed)

    return batch_image, batch_label


# Build dataset
train_data = (tf.data.Dataset.from_tensor_slices((X_Train, Y_Train))
                .map(augmentation, num_parallel_calls=tf.data.AUTOTUNE)
                .repeat(5)
                .shuffle(1024)
                .batch(512)
                .prefetch(tf.data.AUTOTUNE))

val_data = tf.data.Dataset.from_tensor_slices((X_Val, Y_Val)).batch(256).prefetch(tf.data.AUTOTUNE).cache()
test_data = tf.data.Dataset.from_tensor_slices((X_Test, Y_Test)).batch(256).prefetch(tf.data.AUTOTUNE)

preprocessing_layer = tf.keras.Sequential([
    tf.keras.layers.Resizing(height=32, width=32),
    tf.keras.layers.Rescaling(1./255)])

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip("horizontal_and_vertical"),
  tf.keras.layers.RandomRotation(0.2)
])

model = tf.keras.Sequential([
    tf.keras.Input(shape=(32, 32, 3)),

    tf.keras.layers.Conv2D(64, 3, padding="valid"),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, 3, padding="valid"),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(256, 3, padding="valid"),
    tf.keras.layers.Activation("relu"),

    tf.keras.layers.GlobalMaxPool2D(),

    # no drop after convs
    #0.8 74
    #0.6 0.75
    #0.4 77
    #0.25 77
    # no dropout
    tf.keras.layers.Dense(1024, activation="relu"),
    tf.keras.layers.Dropout(rate=0.4),
    tf.keras.layers.Dense(1024, activation="relu"),
    tf.keras.layers.Dropout(rate=0.4),

    tf.keras.layers.Dense(100, activation="softmax")
])

top_10_accuracy = tf.keras.metrics.SparseTopKCategoricalAccuracy(
    k=10, name='sparse_top_k_categorical_accuracy', dtype=None
)

model.compile(loss="sparse_categorical_crossentropy",
              metrics=["sparse_categorical_accuracy", top_10_accuracy],
              optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))

print(model.summary())


def scheduler(epoch, lr):
    if epoch < 15:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


# Define callbacks
early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=25)
save_best_model = tf.keras.callbacks.ModelCheckpoint(filepath="best_model", save_only_best=True,
                                                     save_weights_only=True, verbose=1)
reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)

model.fit(train_data, validation_data=val_data, epochs=500, callbacks=[save_best_model, early_stop_callback])

model.load_weights("best_model")

print(f"Evaluation metrics last model: {model.evaluate(test_data)}")

model.load_weights("best_model")
print(f"Evaluation metrics best model's weights: {model.evaluate(test_data)}")

model.save(filepath="cfar100_model", save_format="h5")
